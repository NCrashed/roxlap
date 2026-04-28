//! Shared little-endian byte cursor for the format parsers.
//!
//! Returns a generic [`OutOfBounds`] error that each format converts
//! into its own `ParseError::Truncated` variant via `From`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutOfBounds {
    pub at: usize,
    pub need: usize,
}

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8, OutOfBounds> {
        let buf = self.read_bytes(1)?;
        Ok(buf[0])
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16, OutOfBounds> {
        let buf = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([buf[0], buf[1]]))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, OutOfBounds> {
        let buf = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]))
    }

    pub(crate) fn read_i16(&mut self) -> Result<i16, OutOfBounds> {
        let buf = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([buf[0], buf[1]]))
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32, OutOfBounds> {
        let buf = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]))
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32, OutOfBounds> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub(crate) fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], OutOfBounds> {
        let end = self.pos.checked_add(n).ok_or(OutOfBounds {
            at: self.pos,
            need: n,
        })?;
        if end > self.bytes.len() {
            return Err(OutOfBounds {
                at: self.pos,
                need: n,
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    pub(crate) fn peek(&self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.bytes.len() {
            return None;
        }
        Some(&self.bytes[self.pos..end])
    }
}
