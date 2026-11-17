//! Metadata filters over JSON payloads.
//!
//! Filters are JSON:
//!
//! ```json
//! {"field": "color", "eq": "red"}
//! {"field": "price", "range": {"gte": 10, "lt": 20}}
//! {"and": [f1, f2]}   {"or": [f1, f2]}   {"not": f}
//! ```
//!
//! Each segment keeps a `PayloadIndex`: a bitmap per (field, string/bool value)
//! and a sorted (value, id) list per numeric field. Array fields index each
//! element, so `{"tags": ["a", "b"]}` matches `{"field": "tags", "eq": "a"}`.
//! A filter evaluates to a bitmap of matching internal ids.

use std::collections::HashMap;

use serde_json::Value;

use crate::bitset::Bitset;

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    Eq { field: String, value: Scalar },
    Range { field: String, min: Bound, max: Bound },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Scalar {
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    Unbounded,
    Inclusive(f64),
    Exclusive(f64),
}

impl Filter {
    pub fn parse(json: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        Self::from_json(&v)
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        let obj = v.as_object().ok_or("filter must be an object")?;
        let list = |key: &str| -> Result<Vec<Filter>, String> {
            obj[key]
                .as_array()
                .ok_or(format!("{key:?} takes a list of filters"))?
                .iter()
                .map(Self::from_json)
                .collect()
        };
        if obj.contains_key("and") {
            return Ok(Filter::And(list("and")?));
        }
        if obj.contains_key("or") {
            return Ok(Filter::Or(list("or")?));
        }
        if let Some(inner) = obj.get("not") {
            return Ok(Filter::Not(Box::new(Self::from_json(inner)?)));
        }

        let field = obj
            .get("field")
            .and_then(Value::as_str)
            .ok_or("filter needs \"and\", \"or\", \"not\" or a \"field\"")?
            .to_string();
        if let Some(value) = obj.get("eq") {
            return match value {
                Value::String(s) => Ok(Filter::Eq {
                    field,
                    value: Scalar::Str(s.clone()),
                }),
                Value::Bool(b) => Ok(Filter::Eq {
                    field,
                    value: Scalar::Bool(*b),
                }),
                Value::Number(n) => {
                    let x = n.as_f64().ok_or("number out of range")?;
                    Ok(Filter::Range {
                        field,
                        min: Bound::Inclusive(x),
                        max: Bound::Inclusive(x),
                    })
                }
                _ => Err(format!("{field}: eq takes a string, bool or number")),
            };
        }
        if let Some(range) = obj.get("range") {
            let range = range.as_object().ok_or("range must be an object")?;
            let num = |key: &str| -> Result<Option<f64>, String> {
                range
                    .get(key)
                    .map(|v| v.as_f64().ok_or(format!("{field}: range.{key} must be a number")))
                    .transpose()
            };
            let min = match (num("gt")?, num("gte")?) {
                (Some(x), _) => Bound::Exclusive(x),
                (None, Some(x)) => Bound::Inclusive(x),
                (None, None) => Bound::Unbounded,
            };
            let max = match (num("lt")?, num("lte")?) {
                (Some(x), _) => Bound::Exclusive(x),
                (None, Some(x)) => Bound::Inclusive(x),
                (None, None) => Bound::Unbounded,
            };
            return Ok(Filter::Range { field, min, max });
        }
        Err(format!("{field}: filter needs \"eq\" or \"range\""))
    }
}

/// Checks a payload is empty or a JSON object, the only shapes we index.
pub(crate) fn validate_payload(payload: &[u8]) -> Result<(), String> {
    if payload.is_empty() {
        return Ok(());
    }
    match serde_json::from_slice::<Value>(payload) {
        Ok(Value::Object(_)) => Ok(()),
        Ok(_) => Err("payload must be a JSON object".into()),
        Err(e) => Err(format!("payload is not valid JSON: {e}")),
    }
}

#[derive(Default)]
pub(crate) struct PayloadIndex {
    eq: HashMap<String, HashMap<Scalar, Bitset>>,
    nums: HashMap<String, Vec<(f64, u32)>>,
    /// Numeric lists are sorted (segments); otherwise range scans are linear (memtable).
    sorted: bool,
}

impl PayloadIndex {
    pub fn build<'a>(payloads: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut index = Self::default();
        for (id, p) in payloads.into_iter().enumerate() {
            index.add(id as u32, p);
        }
        index.finish();
        index
    }

    pub fn add(&mut self, id: u32, payload: &[u8]) {
        let Ok(Value::Object(obj)) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        for (key, value) in &obj {
            self.add_value(id, key, value, true);
        }
        self.sorted = false;
    }

    fn add_value(&mut self, id: u32, key: &str, value: &Value, top: bool) {
        let mut eq = |s: Scalar| {
            self.eq
                .entry(key.to_string())
                .or_default()
                .entry(s)
                .or_default()
                .set(id as usize)
        };
        match value {
            Value::String(s) => eq(Scalar::Str(s.clone())),
            Value::Bool(b) => eq(Scalar::Bool(*b)),
            Value::Number(n) => {
                if let Some(x) = n.as_f64() {
                    self.nums.entry(key.to_string()).or_default().push((x, id));
                }
            }
            Value::Array(items) if top => {
                for item in items {
                    self.add_value(id, key, item, false);
                }
            }
            _ => {}
        }
    }

    /// Sorts numeric lists so range filters can binary search.
    pub fn finish(&mut self) {
        for list in self.nums.values_mut() {
            list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        }
        self.sorted = true;
    }

    /// Ids in `0..n` matching `filter`.
    pub fn eval(&self, filter: &Filter, n: usize) -> Bitset {
        match filter {
            Filter::Eq { field, value } => self
                .eq
                .get(field)
                .and_then(|m| m.get(value))
                .cloned()
                .unwrap_or_default(),
            Filter::Range { field, min, max } => {
                let mut out = Bitset::default();
                let Some(list) = self.nums.get(field) else {
                    return out;
                };
                let above = |x: f64| match *min {
                    Bound::Unbounded => true,
                    Bound::Inclusive(m) => x >= m,
                    Bound::Exclusive(m) => x > m,
                };
                let below = |x: f64| match *max {
                    Bound::Unbounded => true,
                    Bound::Inclusive(m) => x <= m,
                    Bound::Exclusive(m) => x < m,
                };
                if self.sorted {
                    let start = list.partition_point(|&(x, _)| !above(x));
                    for &(x, id) in &list[start..] {
                        if !below(x) {
                            break;
                        }
                        out.set(id as usize);
                    }
                } else {
                    for &(x, id) in list {
                        if above(x) && below(x) {
                            out.set(id as usize);
                        }
                    }
                }
                out
            }
            Filter::And(fs) => {
                let mut it = fs.iter();
                let Some(first) = it.next() else {
                    return Bitset::full(n);
                };
                it.fold(self.eval(first, n), |acc, f| acc.and(&self.eval(f, n)))
            }
            Filter::Or(fs) => fs
                .iter()
                .fold(Bitset::default(), |acc, f| acc.or(&self.eval(f, n))),
            Filter::Not(f) => Bitset::full(n).and_not(&self.eval(f, n)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(sorted: bool) -> PayloadIndex {
        let payloads = [
            r#"{"color": "red", "price": 10, "tags": ["a", "b"], "sale": true}"#,
            r#"{"color": "blue", "price": 25.5, "tags": ["b"]}"#,
            r#"{"color": "red", "price": 30, "sale": false}"#,
            r#"not json"#,
            r#"{"price": -1, "nested": {"color": "red"}}"#,
        ];
        let mut idx = PayloadIndex::default();
        for (i, p) in payloads.iter().enumerate() {
            idx.add(i as u32, p.as_bytes());
        }
        if sorted {
            idx.finish();
        }
        idx
    }

    fn ids(idx: &PayloadIndex, filter: &str) -> Vec<u32> {
        idx.eval(&Filter::parse(filter).unwrap(), 5).iter().collect()
    }

    #[test]
    fn evaluates_filters_sorted_and_unsorted() {
        for sorted in [false, true] {
            let idx = index(sorted);
            assert_eq!(ids(&idx, r#"{"field": "color", "eq": "red"}"#), vec![0, 2]);
            assert_eq!(ids(&idx, r#"{"field": "tags", "eq": "b"}"#), vec![0, 1]);
            assert_eq!(ids(&idx, r#"{"field": "sale", "eq": true}"#), vec![0]);
            assert_eq!(ids(&idx, r#"{"field": "price", "eq": 30}"#), vec![2]);
            assert_eq!(
                ids(&idx, r#"{"field": "price", "range": {"gte": 10, "lt": 30}}"#),
                vec![0, 1]
            );
            assert_eq!(
                ids(&idx, r#"{"field": "price", "range": {"gt": 10}}"#),
                vec![1, 2]
            );
            assert_eq!(ids(&idx, r#"{"field": "price", "range": {"lte": 0}}"#), vec![4]);
            assert_eq!(
                ids(
                    &idx,
                    r#"{"and": [{"field": "color", "eq": "red"}, {"field": "price", "range": {"gt": 15}}]}"#
                ),
                vec![2]
            );
            assert_eq!(
                ids(
                    &idx,
                    r#"{"or": [{"field": "color", "eq": "blue"}, {"field": "sale", "eq": false}]}"#
                ),
                vec![1, 2]
            );
            assert_eq!(
                ids(&idx, r#"{"not": {"field": "color", "eq": "red"}}"#),
                vec![1, 3, 4]
            );
            assert_eq!(ids(&idx, r#"{"field": "missing", "eq": "x"}"#), Vec::<u32>::new());
            assert_eq!(ids(&idx, r#"{"and": []}"#), vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn rejects_malformed_filters() {
        for bad in [
            r#"[]"#,
            r#"{"field": "x"}"#,
            r#"{"eq": "x"}"#,
            r#"{"field": "x", "eq": [1]}"#,
            r#"{"field": "x", "range": {"gt": "a"}}"#,
            r#"{"and": {"field": "x", "eq": 1}}"#,
        ] {
            assert!(Filter::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn payload_validation() {
        assert!(validate_payload(b"").is_ok());
        assert!(validate_payload(br#"{"a": 1}"#).is_ok());
        assert!(validate_payload(b"[1, 2]").is_err());
        assert!(validate_payload(b"{oops").is_err());
    }
}
