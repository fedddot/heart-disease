# Introduction

This study aims to explore the possibilities of statistical analysis and machine learning methods in the diagnosis of heart disease. Within the study, we will work with available data on almost three hundred patients (laboratory measurements, anamnesis, diagnosis) and try to find a functional relationship between these data and the diagnosis.

## Study plan

Within this study, we will go through the full cycle of working with data. We will load a raw dataset, clean it from incorrect and irrelevant information, analyze the statistical significance of each parameter in making predictions, and apply several interesting data dimensionality reduction methods. Finally, we will build several machine learning models to analyze the data, test them and compare the results.<br>

Mathematically speaking, we will try to establish a relationship of the form (1), expressing the probability $Pr$ that a given patient with parameters (measurements, anamnesis) $X$ has a heart disease ($y = 1$).

$$
\begin{equation}
   f(X) = Pr(y = 1 | X)
\tag{1}
\end{equation}
$$

Contents:

```{tableofcontents}
```
