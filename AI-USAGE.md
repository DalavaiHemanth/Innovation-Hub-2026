\# AI Usage



\## AI tools used



ChatGPT was used during development as a coding and reasoning assistant.



It was used for:



\- Reviewing the challenge requirements and submission structure.

\- Discussing feature engineering ideas for the gateway failure prediction task.

\- Reviewing and improving the Random Forest modelling approach.

\- Checking validation methodology, including gateway-held-out validation.

\- Reviewing prediction and submission validation logic.

\- Helping identify reproducibility and one-command execution requirements.

\- Reviewing documentation and preparing the final submission structure.



All generated code and analysis were run and checked locally against the provided challenge data.



\## Example of AI output that was checked and rejected



During the analysis, an initial cost backtest produced a misleading result because the available field-visit records did not cover the complete eight-week challenge prediction period.



The initial interpretation could have treated the absence of observed visits as zero operational cost. After checking the date coverage of the field-visit data, this was rejected because the available observations ended before the full challenge window.



Therefore, no unsupported claim of zero cost or a complete eight-week operational cost improvement is made in the final submission.



\## Human verification



The final pipeline was executed locally using:



&#x20;   python run.py



The pipeline successfully:



1\. Built the ML features.

2\. Trained the Random Forest.

3\. Generated predictions for all eight required weeks.

4\. Selected exactly 15 gateways per week.

5\. Passed `validate\_submission.py`.



The final model and predictions were therefore checked using the supplied challenge data and validation script rather than being accepted solely from AI-generated suggestions.

