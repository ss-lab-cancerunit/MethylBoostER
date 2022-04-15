# MethylBoostER - XGBoost to classify kidney cancer subtypes


MethylBoostER is a multiclass XGBoost model that can classify pathological sub-types of renal tumours (ie, kidney cancer subtypes). This repository includes the code, the trained models and all kinds of results that demonstrate MethylBoostER's performance.

### Explanation of the files


Look at the data in data/ folder:

- See `UMAPs_of_data.ipynb` and `UMAPs/` for UMAP visualisations of the data
- See `combining_EPIC_and_450k.R` (and perhaps `preprocess_TCGA_data.Rmd`) for how we processed the training/testing set
- See `preprocess_validation_data.Rmd` for the preprocessing of the validation datasets 
- The processed data we used are stored in this data repository: https://zenodo.org/record/6463893 (DOI 10.5281/zenodo.6463892)
    + See the README within this repo for information about this data

Train the XGBoost model:

- Run `xgboost.ipynb` (or `xgboost.py`)


See the testing set results:

- Look in `figs_xgboost/`. 
    - Confusion matrices, ROC curves and precision recall curves are in `figs/`
    - Trained XGBoost models are in `xgboost_models/`
    - Testing set scores and metrics are in `scores.json` (you can read it using pickle)
    - Predictions for all samples are in `all_test_predictions`
- Look at `xgboost_results.ipynb`. This loads the saved models and re-generates the saved results figures and metrics.
    - Also generates sample number barplots for the training/testing set
    
Evaluate on external datasets:

- Run `xgboost_external_validation.ipynb` and/or look in `figs_xgboost_validation/`


Get high and moderate confidence predictions:

- Look at `high_moderate_conf_predictions.ipynb` and/or `figs_xgboost_high_mod_conf/`


Evaluate on multiregion samples:

- Run `xgboost_multiregion_predictions.ipynb`
- See `figs_xgboost/figs/multiregion_predictions.svg` and `figs_xgboost_validation/evelonn_multiregion_predictions.svg` figures


Find the features the trained XGBoost models are using:

- Look at `xgboost_important_features.ipynb`
- Also see `map_features_to_genes.Rmd`


Analyse these features:

- See `feature_analysis/`:
    - For gene list enrichment results, see `Fishers tests.ipynb` (and `fdr_correction.R`)
    - For visualisation of the 38 features found in all 4 models, see `important_probe_plots.ipynb`
    - For GREAT analysis (GO enrichment for genomic regions), see `GREAT_analysis.Rmd`
    - For the genomic location of the features, see `Feature_analysis.Rmd` (and folder `gat/` for the GAT analysis results)
    - For class-specific feature analysis, see `Class_specific_feature_analysis.ipynb`, `Class_specific_feature_analysis.Rmd` and `features_per_class/` (the results folder)


Look at a model trained only to classify chRCC and oncocytoma:

- See `onc_vs_chRCC_experiment/` folder


Find out what happens when we vary XGBoost's max_tree_depth parameter:

- See `Tree_depth_experiment.ipynb` and `tree_depth_experiment/` folder.



### How do I find the code to generate Figure x?
- 2a: `xgboost_results.ipynb`
- 2b: `data/UMAPs_of_data.ipynb`
- 2c: `xgboost_results.ipynb`
- 2d: `data/UMAPs_of_data.ipynb`
- 2e: `xgboost_results.ipynb`
- 3a: `high_moderate_conf_predictions.ipynb`
- 3b: `high_moderate_conf_predictions.ipynb`
- 4a: `xgboost_external_validation.ipynb`
- 4b: `high_moderate_conf_predictions.ipynb`
- 4c,d,e and f: `high_moderate_conf_predictions.ipynb` and `xgboost_external_validation.ipynb`
- 5: `xgboost_multiregion_predictions.ipynb`
- 7a: `feature_analysis/Feature_analysis.Rmd`
- 7c: `feature_analysis/GREAT_analysis.Rmd`

- S1: `xgboost_results.ipynb`
- S2: `data/UMAPs_of_data.ipynb`
- S3: `high_moderate_conf_predictions.ipynb`
- S4a: `xgboost_external_validation.ipynb`
- S4b: `high_moderate_conf_predictions.ipynb`
- S4c: `high_moderate_conf_predictions.ipynb`
- S5: `xgboost_external_validation.ipynb`
- S6: `onc_vs_chRCC_experiment/onc_vs_chRCC_hyperparameter_tuning.ipynb`
- S7: `data/UMAPs_of_data.ipynb`
- S9: `feature_analysis/important_probe_plots.ipynb`
- S11a: `feature_analysis/Class_specific_feature_analysis.Rmd`
- S11b, c, d and e: `feature_analysis/Class_specific_feature_analysis.Rmd`
- S11f: `feature_analysis/Class_specific_feature_analysis.ipynb`
- S12a: `data/UMAPs_of_data.ipynb`
- S12b: `data/plot_PCA_of_KIRC_expression.R`
- S13: `Tree_depth_experiment.ipynb`


<br/>


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

