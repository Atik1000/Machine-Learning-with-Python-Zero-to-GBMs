# Python-for-Data-Science-and-Machine-Learning-Bootcamp

I will only post the iPython notebooks and they will be posted as I finish them myself. This helps me to keep a track of my studies
and also refer to it on the fly while I am at work. Reference: 

1.https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview

2.https://jovian.ai/

3.https://github.com/codebasics/roadmaps/blob/master/machine-learning-engineer-roadmap-2021/ml_engineer_roadmap_2021.md

4.https://www.youtube.com/watch?v=3FG-Fpvzymo&list=PLAoF4o7zqskR7U98D799FKHkZ4YrHKPqs&index=1

5.Books :https://bcebakhtiyarpur.org/wp-content/uploads/2020/03/file_5e748501c810d.pdf

6.If you know bangla then https://kazis-organization.gitbook.io/undefined-1

Following are the topics posted-

1) Python Crash Course
2) NumPy (Numeric Python)
3) Pandas
4) Matplotlib
5) Seaborn
6) Pandas Built-in Data Visualization
7) Plotly and Cufflinks
8) Geograhical Plotting
9) Data - Capstone Project
10) Linear Regression
11) Logistic Regression
12) K-Nearest Neighbors
13) Decision Trees and Random Forests
14) Support Vector Machines
15) K-Means Clustering
16) Principal Component Analysis


# what is machine -learning-mean
As a human wha we will do something understand --

1)Collect Data
2)Understanding data and make mathematical formulas
3)predict future

so if we will train a machine then it will work as a human that is called machine -learning.

# Types of Machine-learning

there are 3 types of Machine-learning methods----


<Table>
  <tr>
        <td>
            <h2><b> supervised machine learning: :</b></h2>
            <p>One practical example of supervised learning problems is predicting house prices. How is this achieved? First, we need data about the houses: square footage, number of rooms, features, whether a house has a garden or not, and so on. We then need to know the prices of these houses, i.e. the corresponding labels.</p>
                <img src="https://neurospace.io/blog/2020/08/what-is-supervised-learning/images/what-is-supervised-learning.png" alt="array img" width="500px" height="300px">
        </td>
   </tr>
   <tr>
    <td>
    <h2><b> Unsupervised machine learning: :</b></h2>
    <p>Unsupervised Learning areas of application include market basket analysis, semantic clustering, recommender systems, etc. The most commonly used Supervised Learning algorithms are decision tree, logistic regression, linear regression, support vector machine</p>
         <img src="https://static.javatpoint.com/tutorial/machine-learning/images/unsupervised-machine-learning-1.png" alt="string img" width="500px" height="300px">
    </td>
    </tr>
    <tr>
            <td>
                <h2><b> Reinforcement machine learning:</b></h2>
                <p> Some of the autonomous driving tasks where reinforcement learning could be applied include trajectory optimization, motion planning, dynamic pathing, controller optimization, and scenario-based learning policies for highways. For example, parking can be achieved by learning automatic parking policies</p>
                    <img src="https://www.guru99.com/images/1/082319_0514_Reinforceme2.png" alt="stack img" width="500px" height="370px">
            </td>
            </td>
    </tr>
</Table>


# Here we will have an understanding equation of a line slope-intercept for machine learning

1) Equation of a line
2) Slope
3) Intercept

### We all know from our high school mathematics equation of a line is ax+by+c=0--------

eg: 2x+5y+6=0.

a=2,b=5,c=6

a,b,c are constants.

The equation of a line can be given in slope-intercept form.

y=Mx+Z

where M = slope

Z=intercept

Intercept: Intercept value is the value at which the line bisects the y-coordinate.

Let's take an example.

Take two lines
l1: y=10x+2

l2: y=20x+2

for both the lines intercept value is 2. ie both the lines will bisect the y coordinate at 2.

Slope: The value of the slope gives us the slant and direction of the line.

To understand how the slope value gives slant and direction.

First, we have to understand how to calculate the slope given 2 points.

Take any two points on a line.

Lets say (x1,y1) and (x2,y2).

The equation to calculate the slope of the line is given as:

slope = y2-y1/x2-x1

slope = change in y/change in x

(x1,y1)=(2,8)

(x2,y2)=(6,9)

slope=9-8/6-2

slope = .25

As slope value increases slantness of the line increases.

Let's check this with an example.
L1, L2 lines bisect at a point. Say that point as (x1,y1).

Observe from the figure above the L2 line is more slant when compared to the L1 line.

So the slope of the l2 line should be more.

Take a point (x2,y2) on L1

Take a point (x2,y5) on L2

So x2-x1 value is equal for both the lines.

The denominator is equal for both the lines.

Observe from figure y5-y1 is large when compared to y2-y1.

For the same change in x2-x1 change in y5-y1 is more when compared to change in y2-y1.

The reason is the L2 line is slant than the L1 line.

You can observe the difference clearly in the above figure.

The sign of the slope gives the direction of the line.

Let's understand this with an example.
It has two lines L1and L2.

Take two points on l1. As the x value increases y value decreasing.

Observe this from the figure.

The points are (2.5,2.5) and (3.5,1.9)

x increases from 2.5 to 3.5

y decreases from 2.5 to 1.9

As x increasing y value decreasing the line is moving down.

If y decreases as x increases. From the equation of slope, y2-y1 ie y2 is small when compared to y1.

So 1.9-2.5 =-0.6.

we get a negative slope.

Look at the line l2.

As x value increases y value increases. So we get a positive slope.

Whenever we have a positive slope line moving up.

The understanding of slope, intercept will help a lot in understanding machine learning algorithms.