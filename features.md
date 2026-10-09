You are working on my existing Avionics Anomaly Detection project. I have exactly two objectives:

1. **Make the UI significantly better.**
2. **Make the existing code work correctly and reliably.**

Do not expand the project scope.

### 1. Improve the UI
- Make the Streamlit dashboard look polished, professional, modern, and suitable for a research demonstration.
- Improve the layout, visual hierarchy, spacing, typography, charts, metric cards, tables, and sidebar.
- Make the workflow intuitive: configure the dataset and model, run detection, and understand the results.
- Clearly communicate loading states, errors, warnings, and empty states.
- Ensure the UI remains usable at different screen sizes.
- Preserve all existing functionality and useful information. Do not sacrifice correctness for visual appearance.

### 2. Make the code work
- Inspect the entire existing codebase before making changes.
- Trace the main execution paths, identify actual bugs, and fix them at their source.
- Verify dataset loading, preprocessing, model selection, parameter handling, model fitting, anomaly scoring, thresholding, evaluation metrics, and result visualization.
- Check edge cases, invalid inputs, missing data, numerical issues, and model-specific failures.
- Ensure changing the configuration cannot accidentally display stale results from a previous run.
- Check that metrics, plots, and explanations are consistent with the actual results.
- Fix dependency, import, configuration, and execution issues where necessary.
- Add or update tests where useful. Run the available tests and perform an end-to-end smoke test of the application.
- Do not hide exceptions, fabricate successful results, or silently skip failing functionality.

### Constraints
- Work with the existing architecture wherever practical.
- Do not rewrite working components just for stylistic reasons.
- Do not add unrelated features, new models, unnecessary dependencies, or infrastructure.
- Do not change the project's research objectives or make unsupported claims about model performance.
- Do not stop after producing a list of recommendations. Actually implement the necessary fixes and UI improvements in the codebase.
- Do not claim that something works unless you have verified it. Clearly report anything that could not be tested.

### Completion criteria
The task is complete when:
1. The application starts using documented instructions.
2. The primary workflow runs successfully from configuration through displaying results.
3. Existing models and supported datasets work, or any genuine blockers are explicitly documented.
4. The UI is visibly more polished and easier to use.
5. Tests and smoke checks have been run, with failures fixed where possible.
6. The final response summarizes the changes, files modified, verification performed, and any remaining issues.

**Prioritize these two objectives above everything else: improve the UI and make the existing code work.** Inspect first, implement, test, and iterate.