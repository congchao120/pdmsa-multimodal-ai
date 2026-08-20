# Classification data organization

No clinical data are distributed with this repository. `example_manifest.csv` contains synthetic,
path-shaped records only; it does not include images or real subject identifiers.

The classifier reads **pre-fused RGB PNG files**, not raw MRI/PET volumes. Create a private
manifest with one row per subject and dynamically selected axial slice:

```csv
subject_id,label,center_slice_index,slice_offset,slice_index,fused_path
SUBJ0001,1,18,-2,16,private/fused/SUBJ0001_slice016.png
SUBJ0001,1,18,-1,17,private/fused/SUBJ0001_slice017.png
SUBJ0001,1,18,0,18,private/fused/SUBJ0001_slice018.png
```

Required fields:

- `subject_id`: immutable pseudonym; never a name, medical-record number, accession number,
  DICOM identifier, or server-side primary key;
- `label`: `0` for MSA-P and `1` for PD under the source-code convention;
- `center_slice_index`: zero-based index of the axial slice with the largest segmented ROI area;
- `slice_offset`: exactly one of `-2, -1, 0, 1, 2` for every subject;
- `slice_index`: zero-based absolute index, equal to `center_slice_index + slice_offset`;
- `fused_path`: path to the corresponding three-channel RGB PNG. Relative paths are resolved
  against `data.root` in the classification configuration.

The preprocessing mask must be on the canonical axial grid before selecting along axis 2. The
public selector uses the maximum striatal-mask area as offset 0 and requires two real neighbors on
both sides; edge maxima are rejected rather than shifted or duplicated.

All five rows for one subject must have the same label, center, and exactly one row per relative
offset. Different subjects may have different center and absolute slice indices. File names are
opaque and must not be used to infer zero- or one-based indexing. The
RGB channel-to-modality mapping (for example FDG/CFT/T2WI) belongs in the experiment's
preprocessing record; it cannot be inferred reliably from RGB values.

Do not pair images by directory listing order or reconstruct subject identity from diagnosis alone.
Every join must use an explicitly verified pseudonymous `subject_id` and `slice_offset`. When a
record cannot be linked deterministically, assign a new provisional pseudonym and keep it out of
cross-experiment merges until documentary evidence resolves the match.

Run the manifest audit before creating folds:

```bash
pdmsa-audit-manifest --config configs/classification.toml
pdmsa-audit-manifest --config configs/classification.toml --check-files
```

The first command validates schema, labels, exact center-minus/plus-two mapping, and five-slice
completeness. The second additionally
checks local files and should be run only where the private images are mounted.
