import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp

class explore:
    def __init__(self):
        self.self = self

    def set_cap_floor(self, column, hard_floor=None, hard_cap=None):
        '''
        Takes DF column. Caps values at +/- 2.326 st. dev. away from mean. 
        Returns new column with capped and floored values. 
        '''
        stdev = column.std()
        mean = column.mean()
        cap = mean + (2.326 * stdev) if hard_cap is None else hard_cap
        floor = mean - (2.326 * stdev) if hard_floor is None else hard_floor
        total_above = np.sum(column > cap)
        total_below = np.sum(column < floor)
        print('-----\nCol Name: {}\nTotal values: {}\nFloor: {:.03f} ({} values; {:.02f}%)\nCap: {:.03f} ({} values; {:.02f}%)'.format(
            column.name, len(column), floor, total_below, (total_below / len(column)) * 100, 
            cap, total_above, (total_above / len(column)) * 100
        )
            )
        return column.clip(lower = floor, upper = cap)

    def compare_hists(self, df, target_col='', bins=10, specific_cols=None, list_of_labels=None, cap_floor=None, hard_floor=None, hard_cap=None):
        '''
        Plots normalized and raw histograms next to each other. Caps values at +/- 2 st. dev. from mean.
        Target or class column must be specific. Can be multi-class as well. 
        
        - df: DataFrame
        - target_col: String
        - bins: int of bins to split data by
        - specific_cols: list of str
        - list_of_labels: list of str
        '''
        df_use = None
        if specific_cols:
            specific_cols.append(target_col)
            df_use = df[specific_cols]
        else:
            df_use = df

        for col in df_use.columns:
            if col == target_col:
                continue
            else:
                #Lets cap and floor the outliers so that we can see pretty distributions
                df_use[col] = df_use[col] if cap_floor is None else self.set_cap_floor(df_use[col])
                _, all_bins = np.histogram(df_use[col], bins=10)

                # Create a normalized histogram for each class
                target_array = [df_use[df_use[target_col] == target_value][col] for target_value in df_use[target_col].unique()]
                weights_array = [np.ones_like(x) / x.size for x in target_array]
                class_labels = df_use[target_col].unique() if list_of_labels is None else list_of_labels
                label_array = [target_value for target_value in class_labels]

                plt.figure(figsize=(12,4))

                plt.subplot(1,2,1)
                plt.title('Histogram Comparison: {} (normalized)'.format(col))
                hist_comp = plt.hist(target_array, weights=weights_array, bins=all_bins, label=label_array)
                plt.legend()
                plt.grid(True)

                plt.subplot(1,2,2)
                plt.title('Histogram Comparison: {} (Raw)'.format(col))
                hist_comp_raw = plt.hist(target_array, bins=all_bins, label=label_array)
                plt.legend()
                plt.grid(True)

                plt.show()
                df_pct = pd.DataFrame(data=list(zip(*hist_comp[0])),index=hist_comp[1][:-1].round(2))
                df_raw = pd.DataFrame(data=list(zip(*hist_comp_raw[0])),index=hist_comp_raw[1][:-1].round(2))
                print('Raw Hist:\n{}\n'.format(df_raw))
                print('Norm Hist:\n{}\n'.format(df_pct.round(3)))
                print('Value Counts from DF:\n{}\n'.format(df_use[target_col].value_counts()))
                print('Value Counts from Hist:\n{}\n'.format(df_raw.sum()))

class model:
    def __init__(self):
        self.self = self

    def optimal_cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value

        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold'])

    ## Create method to cross val models and produce scores:
    def cross_val_models(self, est_list, est_labels, xval, yval, cv_use = 3, cv_type='clf', custom_scoring=None):
        '''
        Parameters
        ----------
        est_list : list of estimators to perform cross-validation on

        est_labels : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value
        '''
        print('{}-fold CV'.format(cv_use))
        

        if cv_type == 'clf':
            scoring_array = ['roc_auc','accuracy','precision','recall'] if custom_scoring is None else custom_scoring
        
        elif cv_type == 'regr' and not custom_scoring:
            scoring_array = ['r2_score','rmse'] if custom_scoring is None else custom_scoring
        

        for est, label in zip(est_list, est_labels):
            for score_type in scoring_array:

                score_val = cross_val_score(estimator = est, X = xval, y = yval, scoring = score_type, cv = cv_use)
                print('{}: {:.03f} (+/-{:.03f}) [{}]'.format(score_type, score_val.mean(), score_val.std(), label))
            
            print('-----\n')

    def cross_val_roc_auc_plot(self, est_list, xval, yval, cvs_use=5):
        for est in est_list:
            # Classification and ROC analysis
            # Run classifier with cross-validation and plot ROC curves
            
            cv = StratifiedKFold(n_splits=cvs_use)

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            plt.figure(figsize=(12,6))

            i = 0
            for train, test in cv.split(xval, yval):
                probas_ = est.fit(xval.iloc[train], yval.iloc[train]).predict_proba(xval.iloc[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(yval.iloc[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.7,
                        label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

                i += 1
            
            print(est)

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = {:0.3f} $\pm$ {:0.3f})'.format(mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.show()
            print('-----\n')

    def cross_val_precision_recall_auc(self, est_list, xval, yval, cvs_use=5):
        for est in est_list:
            # Classification and ROC analysis
            # Run classifier with cross-validation and plot ROC curves
            
            cv = StratifiedKFold(n_splits=cvs_use)

            mean_precs = []
            mean_auc = []
            
            plt.figure(figsize=(12,6))

            i = 0
            for train, test in cv.split(xval, yval):
                probas_ = est.fit(xval.iloc[train], yval.iloc[train]).predict_proba(xval.iloc[test])[:,1]
                p, r, _ = precision_recall_curve(yval.iloc[test], probas_)
                mean_prec_score = average_precision_score(yval.iloc[test], probas_)
                auc_pr = auc(r, p)
                mean_auc.append(auc_pr)
                mean_precs.append(mean_prec_score)
                
                plt.plot(r, p, lw=2, alpha=0.7,
                                    label='P-R fold %d (AUC = %0.2f)' % (i, auc_pr))
                i+=1
            
            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()
            
            print(est)
            print('Avg. Area Under P-R Curve: {:0.3f} (+/-{:0.3f})'.format(np.mean(mean_auc), np.std(mean_auc)))
            print('Avg. Precision Score: {:0.3f} (+/-{:0.3f})'.format(np.mean(mean_precs), np.std(mean_precs)))
            print('-----\n')
