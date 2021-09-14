# NFL Prediction_REPORT
## What is the problem?
The problem is that: we have a group of data about NFL players, including their heights, weights and positions. We want to analyze the physical builds of such players. To be more specific, the goal is: for any given height and weight of a player, what is the best position for him? 
<br>
## Here is a breif look of data_frame
show below:
<br>
|full_name          |number|position        |height (in)|weight (lb)|date_of_birth|team|sign     |
|-------------------|------|----------------|-----------|-----------|-------------|----|---------|
|Alford, Robert     |23    |cornerback      |70         |186        |11/1/88      |ATL |scorpio  |
|Babineaux, Jonathan|95    |defensive tackle|74         |300        |10/12/81     |ATL |libra    |
|Davis, Dominique   |4     |quarterback     |75         |210        |7/17/89      |ATL |cancer   |
|Goodman, Malliciah |93    |defensive end   |76         |276        |1/4/90       |ATL |capricorn|
|Jackson, Steven    |39    |running back    |74         |240        |7/22/83      |ATL |cancer   |
|Jerry, Peria       |94    |defensive tackle|74         |295        |8/23/84      |ATL |virgo    |
|Maponga, Stansly   |90    |defensive end   |74         |265        |3/5/91       |ATL |pisces   |
|Massaquoi, Jonathan|96    |defensive end   |74         |264        |5/18/88      |ATL |taurus   |
|Matthews, Cliff    |98    |defensive end   |76         |268        |8/5/89       |ATL |leo      |
|McClain, Robert    |27    |cornerback      |69         |195        |7/22/88      |ATL |cancer   |
|Peters, Corey      |91    |defensive tackle|75         |305        |6/8/88       |ATL |gemini   |
|Renfree, Sean      |12    |quarterback     |77         |225        |4/28/90      |ATL |taurus   |
|Rodgers, Jacquizz  |32    |running back    |66         |196        |2/6/90       |ATL |aquarius |

<br>
First, we calculate the mean of height and weight: 
<br>

<table>
  <thead>
    <tr>
      <th>Position</th>
      <th>Mean Height (in)</th>
      <th>Mean Weight (lb)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>defensive tackle</td>
      <td>74.85</td>
      <td>309.77</td>
    </tr>
    <tr>
      <td>quarterback</td>
      <td>75.20</td>
      <td>223.76</td>
    </tr>
    <tr>
      <td>defensive end</td>
      <td>76.03</td>
      <td>283.18</td>
    </tr>
    <tr>
      <td>running back</td>
      <td>70.62</td>
      <td>215.27</td>
    </tr>
    <tr>
      <td>cornerback</td>
      <td>71.41</td>
      <td>193.39</td>
    </tr>
  </tbody>
</table>
The brief code for prediction function:

```python
def prediction(height, weight):
    #reconstruct the data frames
    df_info = df[["height (in)", "weight (lb)"]]
    X = np.array(df_info)
    Y = np.array(df_position)

    #find the correct model:
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, Y)
    LinearDiscriminantAnalysis()
    return(clf.predict([[height, weight]]))
```

## Some examples:
```python
print(prediction(70, 155)) #======>['cornerback']
print(prediction(77, 230)) #======>['quarterback']
print(prediction(90, 330)) #======>['defensive end']
print(prediction(60, 320)) #======>['defensive tackle']
print(prediction(66, 260)) #======>['running back']
```

## Following is the Decision surface of classifier:
![NFL Player.png](https://i.loli.net/2021/03/04/dUh6zcAXVWYgEIJ.png)







