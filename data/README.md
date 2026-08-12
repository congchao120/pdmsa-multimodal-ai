# Classification data organization

No clinical data are distributed with this repository. `example_manifest.csv` contains synthetic,
path-shaped records only; it does not include images or real subject identifiers.

The classifier reads **pre-fused RGB PNG files**, not raw MRI/PET volumes. Create a private
manifest with one row per subject and selected axial layer:

```csv
subject_id,label,slice_index,fused_path
SUBJ0001,1,6,private/fused/SUBJ0001_layer6.png
SUBJ0001,1,7,private/fused/SUBJ0001_layer7.png
```

Required fields:

- `subject_id`: immutable pseudonym; never a name, medical-record number, accession number,
  DICOM identifier, or server-side primary key;
- `label`: `0` for MSA-P and `1` for PD under the source-code convention;
- `slice_index`: one of `6, 7, 8, 9, 10` for the retained five-layer workflow;
- `fused_path`: path to the corresponding three-channel RGB PNG. Relative paths are resolved
  against `data.root` in the classification configuration.

All five rows for one subject must have the same label and exactly one row per slice index. The
RGB channel-to-modality mapping (for example FDG/CFT/T2WI) belongs in the experiment's
preprocessing record; it cannot be inferred reliably from RGB values.

Do not pair images by directory listing order or reconstruct subject identity from diagnosis alone.
Every join must use an explicitly verified pseudonymous `subject_id` and `slice_index`. When a
record cannot be linked deterministically, assign a new provisional pseudonym and keep it out of
cross-experiment merges until documentary evidence resolves the match.

Run the manifest audit before creating folds:

```bash
pdmsa-audit-manifest --config configs/classification.toml
pdmsa-audit-manifest --config configs/classification.toml --check-files
```

The first command validates schema, labels, and five-layer completeness. The second additionally
checks local files and should be run only where the private images are mounted.
