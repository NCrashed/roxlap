//! Parse a `.kv6` voxel sprite and print its dimensions + voxel count.
//!
//! Run from the workspace root (so `assets/coco.kv6` resolves):
//!
//! ```sh
//! cargo run --example parse_kv6 -p roxlap-formats
//! ```

use roxlap_formats::kv6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("assets/coco.kv6")?;
    let sprite = kv6::parse(&bytes)?;
    println!(
        "{} × {} × {} voxel grid, {} solid voxels",
        sprite.xsiz,
        sprite.ysiz,
        sprite.zsiz,
        sprite.voxels.len(),
    );
    println!(
        "pivot: ({:.2}, {:.2}, {:.2})",
        sprite.xpiv, sprite.ypiv, sprite.zpiv,
    );
    println!(
        "trailing palette: {}",
        if sprite.palette.is_some() {
            "present"
        } else {
            "none"
        },
    );
    Ok(())
}
