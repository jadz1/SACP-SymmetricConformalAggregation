# Symmetric Aggregation of Conformity Scores for Efficient Uncertainty Sets

This repository contains the official implementation of **SACP (Symmetric Aggregated Conformal Prediction)**, a method for aggregating conformity scores from multiple predictive models to produce **efficient** and **reliable** uncertainty sets with **exact coverage guarantees**.

**Paper (arXiv):** *Symmetric Aggregation of Conformity Scores for Efficient Uncertainty Sets*  
**Link:** https://arxiv.org/abs/2512.06945

---

## Authors

Nabil Alami¹²\*, Jad Zakharia³\*, Souhaib Ben Taieb¹⁴

## Affiliations

¹ Department of Statistics and Data Science, Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI)  
² CentraleSupélec, Université Paris-Saclay  
³ École des Ponts ParisTech  
⁴ Department of Computer Science, University of Mons  

## Contact

nabil.alami@mbzuai.ac.ae  
jad.zakharia@enpc.fr  
souhaib.bentaieb@mbzuai.ac.ae  

---

## Reproducibility (Experiments)

Regression experiments reported in the paper can be reproduced using the provided notebook:

**Run:** `demo.ipynb`

The notebook runs the full pipeline across multiple OpenML regression datasets and random seeds:
- Train a set of base regressors (Linear, Lasso, RF, MLP, Bayesian Ridge, Boosting, SGD)
- Compute uncertainty sets using:
  - Individual conformal predictors (per model)
  - **SACP (ours)**
  - CR / CM majority vote baselines
  - WAgg
  - CSA
  - SACP++

Results are stored in a pandas DataFrame with **coverage** and **average set length** for each method.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{alami2025symmetricaggregationconformityscores,
      title={Symmetric Aggregation of Conformity Scores for Efficient Uncertainty Sets}, 
      author={Nabil Alami and Jad Zakharia and Souhaib Ben Taieb},
      year={2025},
      eprint={2512.06945},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2512.06945}, 
}
