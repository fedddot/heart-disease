# Data Set Description

The data was collected from Cleveland Clinic Foundation.

:::{note}
The authors of the database have requested that any publications resulting from the use of the data include the 
names of the principal investigator responsible for the data collection
at each institution. They would be:

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
Robert Detrano, M.D., Ph.D.
:::

Description of the data is given in Table 1

## Table 1

<table border = "1">
    <tr>
        <th>Column #</th>
        <th>Column label</th>
        <th>Description</th>
        <th>Data type</th>
    </tr>
    <tr>
        <td>0</td>
        <td>age</td>
        <td>age of patient</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>1</td>
        <td>sex</td>
        <td>0 = female; 1 = male</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>2</td>
        <td>cp</td>
        <td>chest pain type<br> 1 = typical angina<br> 2 = atypical angina<br> 3 = non-anginal pain<br> 4 = asymptomatic</td>
        <td>real number, categorical</td>
    </tr>
    <tr>
        <td>3</td>
        <td>trestbps</td>
        <td>resting blood pressure (in mm Hg)</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>4</td>
        <td>chol</td>
        <td>serum cholesterol in mg/dl</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>5</td>
        <td>fbs</td>
        <td>fasting blood sugar > 120 mg/dl (1 = true; 0 = false)</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>6</td>
        <td>restecg</td>
        <td>resting electrocardiographic results<br> 0 = normal<br> 1 = having ST-T wave abnormality (T wave inversions and/or ST levation or depression of > 0.05 mV)<br> 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria</td>
        <td>real number, categorical</td>
    </tr>
    <tr>
        <td>7</td>
        <td>thalach</td>
        <td>maximum heart rate achieved</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>8</td>
        <td>exang</td>
        <td>exercise induced angina (1 = yes; 0 = no)</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>9</td>
        <td>oldpeak</td>
        <td>ST depression induced by exercise relative to rest</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>10</td>
        <td>slope</td>
        <td>the slope of the peak exercise ST segment<br> 1 = upsloping<br> 2 = flat<br> 3 = downsloping
        </td>
        <td>real number</td>
    </tr>
    <tr>
        <td>11</td>
        <td>ca</td>
        <td>number of major vessels (0-3) colored by flourosopy</td>
        <td>real number</td>
    </tr>
    <tr>
        <td>12</td>
        <td>thal</td>
        <td>thalassemia - inherited blood disorder that causes your body to have less hemoglobin than normal<br>3 = normal
            <br>6 = fixed defect<br>7 = reversable defect</td>
        <td>real number, categorical</td>
    </tr>
    <tr>
        <td>13</td>
        <td>num</td>
        <td>diagnosis of heart disease (angiographic disease status)<br> 0 = less than 50% diameter narrowing<br> 1, 2, 3, 4 = more than 50% diameter narrowing</td>
        <td>real number</td>
    </tr>
</table>