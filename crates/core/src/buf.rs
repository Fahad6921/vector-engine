//! Storage that is either an owned `Vec` or a read-only view into a memory-mapped
//! file. Lets a loaded index be searched straight from the page cache without
//! parsing or copying, and turn into an owned, growable one on first write.

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use memmap2::Mmap;

/// Plain-old-data: no padding, every bit pattern valid, so a value can be viewed
/// directly from file bytes. Files are little-endian; so is every target we build for.
///
/// # Safety
/// Implementors must satisfy the above.
pub(crate) unsafe trait Pod: Sized {}
unsafe impl Pod for u8 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for f32 {}
unsafe impl Pod for AtomicU32 {}

#[cfg(not(target_endian = "little"))]
compile_error!("index files are little-endian; big-endian targets are not supported");

pub(crate) enum Buf<T> {
    Owned(Vec<T>),
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
        _t: PhantomData<T>,
    },
}

impl<T: Pod> Buf<T> {
    /// A view of `len` values of `T` starting at byte `offset` of `map`.
    pub fn mapped(map: Arc<Mmap>, offset: usize, len: usize) -> Self {
        let bytes = len.checked_mul(size_of::<T>()).expect("section size overflows");
        assert!(offset + bytes <= map.len(), "section runs past end of file");
        assert_eq!(
            (map.as_ptr() as usize + offset) % align_of::<T>(),
            0,
            "section is misaligned"
        );
        Buf::Mapped {
            map,
            offset,
            len,
            _t: PhantomData,
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, Buf::Mapped { .. })
    }

    /// Mutable access, copying out of the mapping first if needed.
    pub fn to_mut(&mut self) -> &mut Vec<T> {
        if let Buf::Mapped { len, .. } = self {
            let len = *len;
            let mut owned = Vec::with_capacity(len);
            // SAFETY: T is Pod, so a bitwise copy of `len` initialised values is valid.
            unsafe {
                std::ptr::copy_nonoverlapping(self.as_ptr(), owned.as_mut_ptr(), len);
                owned.set_len(len);
            }
            *self = Buf::Owned(owned);
        }
        match self {
            Buf::Owned(v) => v,
            Buf::Mapped { .. } => unreachable!(),
        }
    }
}

impl<T: Pod> Deref for Buf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Buf::Owned(v) => v,
            Buf::Mapped { map, offset, len, .. } => {
                // SAFETY: bounds and alignment were checked in `mapped`, T is Pod, and
                // the mapping lives as long as `self` through the Arc.
                unsafe { std::slice::from_raw_parts(map.as_ptr().add(*offset).cast::<T>(), *len) }
            }
        }
    }
}

/// The raw bytes of a Pod slice.
pub(crate) fn as_bytes<T: Pod>(s: &[T]) -> &[u8] {
    // SAFETY: T is Pod (no padding), so every byte of the slice is initialised.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
}
