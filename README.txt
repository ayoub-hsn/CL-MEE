CL-MEE — AICSET'26 CAMERA-READY REVISION PACKAGE
================================================

This archive contains the
reproduction code, the Foursquare NYC dataset, the validation-trained scorer
checkpoint, tests, reference outputs, and verification tools.

IMPORTANT FILES
---------------
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

