//! Python bindings: `vecengine.Index`.
//!
//!     import numpy as np, vecengine
//!     index = vecengine.Index(128, metric="l2", m=16, ef_construction=200)
//!     index.add(np.random.rand(10_000, 128).astype(np.float32))
//!     ids, distances = index.search(query, k=10, ef=64)

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use ::vecengine::{Hnsw, HnswParams, Metric, Neighbor};

#[pyclass(name = "Index", module = "vecengine")]
struct Index {
    inner: Hnsw,
}

fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::L2 => "l2",
        Metric::InnerProduct => "ip",
        Metric::Cosine => "cosine",
    }
}

/// Flattens results into (ids, distances) of shape (n, k). Missing slots, when the
/// index holds fewer than k vectors, are id -1 and distance inf.
fn to_arrays(results: &[Vec<Neighbor>], k: usize) -> (Array2<i64>, Array2<f32>) {
    let mut ids = Array2::from_elem((results.len(), k), -1i64);
    let mut dists = Array2::from_elem((results.len(), k), f32::INFINITY);
    for (i, res) in results.iter().enumerate() {
        for (j, n) in res.iter().enumerate() {
            ids[[i, j]] = n.id as i64;
            dists[[i, j]] = n.distance;
        }
    }
    (ids, dists)
}

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (dim, metric = "l2", m = 16, ef_construction = 200, seed = 0x5eed))]
    fn new(dim: usize, metric: &str, m: usize, ef_construction: usize, seed: u64) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be positive"));
        }
        if m < 2 {
            return Err(PyValueError::new_err("m must be at least 2"));
        }
        let metric = Metric::from_name(metric).ok_or_else(|| {
            PyValueError::new_err(format!("unknown metric {metric:?}, use l2, ip or cosine"))
        })?;
        let params = HnswParams {
            m,
            ef_construction,
            seed,
        };
        Ok(Self {
            inner: Hnsw::new(dim, metric, params),
        })
    }

    /// Adds a float32 array of shape (n, dim). Ids are assigned in insertion order.
    fn add(&mut self, py: Python<'_>, data: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let [_, d] = *data.shape() else { unreachable!() };
        if d != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected vectors of dim {}, got {d}",
                self.inner.dim()
            )));
        }
        let data = data.as_array().as_standard_layout().into_owned();
        let flat = data.as_slice().expect("standard layout is contiguous");
        let inner = &mut self.inner;
        py.detach(|| inner.add_batch(flat));
        Ok(())
    }

    /// Searches one query (shape (dim,)) or a batch (shape (n, dim)).
    ///
    /// Returns (ids, distances) with shape (k,) or (n, k). Batches run in parallel.
    #[pyo3(signature = (query, k = 10, ef = 64))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArrayDyn<'py, f32>,
        k: usize,
        ef: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let dim = self.inner.dim();
        let shape = query.shape().to_vec();
        let single = match shape.as_slice() {
            [d] if *d == dim => true,
            [_, d] if *d == dim => false,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "query must have shape ({dim},) or (n, {dim}), got {shape:?}"
                )))
            }
        };
        let q = query.as_array().as_standard_layout().into_owned();
        let q = q.as_slice().expect("standard layout is contiguous");
        let inner = &self.inner;

        if single {
            let res = py.detach(|| inner.search(q, k, ef));
            let (ids, dists) = to_arrays(std::slice::from_ref(&res), k);
            let ids: Bound<'py, PyArray1<i64>> = ids.row(0).to_owned().into_pyarray(py);
            let dists: Bound<'py, PyArray1<f32>> = dists.row(0).to_owned().into_pyarray(py);
            return PyTuple::new(py, [ids.into_any(), dists.into_any()]);
        }

        let res = py.detach(|| inner.search_batch(q, k, ef));
        let (ids, dists) = to_arrays(&res, k);
        let ids: Bound<'py, PyArray2<i64>> = ids.into_pyarray(py);
        let dists: Bound<'py, PyArray2<f32>> = dists.into_pyarray(py);
        PyTuple::new(py, [ids.into_any(), dists.into_any()])
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn metric(&self) -> &'static str {
        metric_name(self.inner.metric())
    }

    fn __repr__(&self) -> String {
        let p = self.inner.params();
        format!(
            "Index(dim={}, metric={:?}, m={}, ef_construction={}, len={})",
            self.inner.dim(),
            metric_name(self.inner.metric()),
            p.m,
            p.ef_construction,
            self.inner.len()
        )
    }
}

#[pymodule]
fn vecengine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    Ok(())
}
