# Classification data organization

No clinical data are distributed with this repository. `example_manifest.csv` contains synthetic,
path-shaped records only; it does not include images or real subject identifiers.

The classifier reads **pre-fused RGB PNG files**, not raw MRI/PET volumes. Create a private
manifest with one row per subject and dynamically selected axial slice:

```csv
subject_id,label,center_slice_index,slice_offset,slice_index,fused_path
SYNTHETIC_001,1,18,-2,16,private/fused/SYNTHETIC_001_slice016.png
SYNTHETIC_001,1,18,-1,17,private/fused/SYNTHETIC_001_slice017.png
SYNTHETIC_001,1,18,0,18,private/fused/SYNTHETIC_001_slice018.png
```

Required fields:

- `subject_id`: immutable pseudonym; never a name, medical-record number, accession number,
  DICOM identifier, or server-side primary key;
- `label`: `0` for MSA-P and `1` for PD;
- `center_slice_index`: zero-based index of the axial slice with the largest segmented ROI area;
- `slice_offset`: exactly one of `-2, -1, 0, 1, 2` for every subject;
- `slice_index`: zero-based absolute index, equal to `center_slice_index + slice_offset`;
- `fused_path`: path to the corresponding three-channel RGB PNG. Relative paths are resolved
  against `data.root` in the classification configuration.

The preprocessing mask must be on the canonical axial grid before selecting along axis 2. The
selector uses the maximum striatal-mask area as offset 0 and requires two real neighbors on
both sides; edge maxima are rejected rather than shifted or duplicated.

All five rows for one subject must have the same label, center, and exactly one row per relative
offset. Different subjects may have different center and absolute slice indices. File names are
opaque and must not be used to infer zero- or one-based indexing. The
RGB channel-to-modality mapping (for example FDG/CFT/T2WI) belongs in the experiment's
preprocessing record; it cannot be inferred reliably from RGB values.

Do not pair images by directory-listing order or infer identity from diagnosis. Join records only
with a verified pseudonymous `subject_id` and `slice_offset`; reject ambiguous matches.

Run the manifest audit before creating folds:

```bash
pdmsa-audit-manifest --config configs/classification.toml
pdmsa-audit-manifest --config configs/classification.toml --check-files
```

The first command validates schema, labels, exact center-minus/plus-two mapping, and five-slice
completeness. The second additionally checks the configured local image paths.
