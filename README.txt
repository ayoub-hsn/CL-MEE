CL-MEE — AICSET'26 CAMERA-READY REVISION PACKAGE
================================================

This archive contains the revised Springer/Overleaf manuscript, figures,
reproduction code, the Foursquare NYC dataset, the validation-trained scorer
checkpoint, tests, reference outputs, verification tools, and reviewer response.

IMPORTANT FILES
---------------
- main.tex / main.pdf: revised Springer manuscript and compiled nine-page PDF.
- figures/architecture.pdf: manuscript architecture figure.
- references.bib, svproc.cls, aliascnt.sty, remreset.sty: LaTeX dependencies.
- reproducibility/run_revision_experiments.py: complete standalone experiment runner.
- reproducibility/foursquare_model_weights.json: validation-fitted scorer checkpoint
  used by default to stabilize the reported reference metrics across platforms.
- reproducibility/dataset_TSMC2014_NYC.txt: supplied 31 MB dataset.
- reproducibility/revision_outputs/: authoritative reference JSON and CSV outputs
  from the author's Windows x64 / Python 3.13.15 run.
- reproducibility/tests/: unit tests.
- reproducibility/verify_package.py: checks dataset hash, JSON/CSV consistency,
  reference metrics, budget invariants, operational diagnostics, runtime reference,
  and the paper's key reported values; it can also verify the nine-page PDF.
- reproducibility/capture_environment.py: captures platform and pip-freeze details.
- reproducibility/RUN_ALL_TESTS_WINDOWS.bat: one-click Windows setup, tests,
  experiment run, and verification.
- RESPONSE_TO_REVIEWERS.md: point-by-point response to both reviewers.
- REVISION_GUIDE_FR.md: French guide.
- SUBMISSION_CHECKLIST.md: final author and CMT checklist.
- PACKAGE_VALIDATION_REPORT.md: record of independent package, experiment, and LaTeX checks.
- FINAL_AUDIT_REPORT.md: consolidated map of both reviewers’ requested revisions and final checks.

WINDOWS QUICK START
-------------------
1. Extract the archive.
2. Open the reproducibility folder.
3. Double-click RUN_ALL_TESTS_WINDOWS.bat.
   It creates .venv, installs pinned requirements, captures the environment,
   runs unit tests, reruns the simulation and Foursquare experiments using the
   supplied scorer checkpoint, and verifies the results against the reference
   outputs and main.tex.
4. The new run is written to reproducibility/verification_results/.
5. Read verification_results/verification_report.txt. All checks should say PASS.

Manual Windows CMD commands (from reproducibility):
  py -3.13 -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -m unittest discover -s tests -v
  .venv\Scripts\python.exe run_revision_experiments.py --dataset dataset_TSMC2014_NYC.txt --output-dir verification_results
  .venv\Scripts\python.exe verify_package.py --results-dir verification_results --dataset dataset_TSMC2014_NYC.txt --paper ..\main.tex --reference-dir revision_outputs --pdf ..\main.pdf --max-pages 9

OPTIONAL WEIGHT REFIT
---------------------
The default run loads the supplied validation-fitted checkpoint. To rerun the
SciPy optimizer, use --refit-weights. Numerical optimization can vary slightly
across operating systems and BLAS/SciPy builds; the checkpoint mode is the
reference mode for reproducing the manuscript's numbers. The refit mode is
provided for auditing the fitting procedure, not to replace the reference run.

OVERLEAF
--------
Upload CLMEE_Overleaf_CameraReady_Final.zip using Overleaf > New Project > Upload
Project. Compile with pdfLaTeX. The project includes main.tex, main.pdf, figures,
Springer class/style files, and references. The dataset and Python code are not
needed to compile the manuscript.

REFERENCE METRICS
-----------------
The corrected reference Windows run reports Foursquare P1 NDCG@5=0.1565238900,
which retains 99.9837763% of RankOnly (reported as 99.984% in the paper). P2
retains 100.1165769% (reported as 100.117%). The exact data are in
reproducibility/revision_outputs/revision_results.json. The latest Windows CMD
runtime record is in reproducibility/runtime_reference_windows.json; the paper
reports 8.76 s candidate generation, 1.90 s feature scoring/top-M, and 0.76 s
evidence/policy evaluation (0.0377 ms per event; 40.1% relative overhead).

ORIGINAL MATERIALS AND SUBMISSION
---------------------------------
The original submitted manuscript, notebook, and legacy results are retained
under original_submission/ for audit. Use main.pdf for the revised manuscript.
Do not upload the full dataset/reproduction ZIP to CMT unless the organizers
explicitly request it. Have all authors/supervisor validate the final science,
author details, references, and conference requirements before submission.
