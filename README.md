# Python-for-Data-Science-and-Machine-Learning-Bootcamp

I will only post the iPython notebooks and they will be posted as I finish them myself. This helps me to keep a track of my studies
and also refer to it on the fly while I am at work. Reference: 

1.https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview

2.https://jovian.ai/

3.https://github.com/codebasics/roadmaps/blob/master/machine-learning-engineer-roadmap-2021/ml_engineer_roadmap_2021.md

4.https://www.youtube.com/watch?v=3FG-Fpvzymo&list=PLAoF4o7zqskR7U98D799FKHkZ4YrHKPqs&index=1

5.Books :https://bcebakhtiyarpur.org/wp-content/uploads/2020/03/file_5e748501c810d.pdf

6.If you know bangla then https://kazis-organization.gitbook.io/undefined-1


# if you have any problem understanding then contact me : atik2000.foysal@gmail.com

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

1) Collect Data
2) Understanding data and make mathematical formulas
3) predict future

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

# Point in a Coordinate System 

Here we discuss the point in a coordinate system in machine learning with an example.

We use this point concept in representing data in machine learning.

Form our high school mathematics we all know about 2-dimensional coordinate space
It has X-axis and Y-axis.

A point in a coordinate system is given as an ordered pair of numbers.

Eg (2,2)

To plot point (2,2) in two-dimensional space move, 2 units on the positive side of the x-axis and move 2 units on the positive side of the Y-axis.

In the above figure, the point is shown with a black dot.

We can use this point concept to represent the machine learning data in coordinate space.

Let's check with an example.

In our last discussion, we used tip prediction data.
In the above data, we have 2 columns. 1) Bill amount 2)  Tip amount

Each column in our dataset is considered as a coordinate in our coordinate space.

To represent the above data we need a 2-dimensional coordinate space.
The dataset has three columns so we represent this data in three-dimensional coordinate space.

Each column is taken as a coordinate in a coordinate system.

A point in a three-dimensional coordinate space is given as (500,6,45).

To represent the above data point we move 500 units on X-axis than 6 units towards the y-axis and 45 units towards the z-axis.
The dataset has n columns. So we can represent the data in N-dimensional coordinate space.

As a human, we can not visualize N-dimensional coordinate space.

We are not providing a diagram for N-dimensional space.

Imagine on your own how an N-dimensional coordinate space looks like.

Imagine How to represent data in N-dimensional space.

This imagination helps you a lot in understanding machine learning algorithms.

Most of the machine learning algorithms use an N-dimensional coordinate system.

So we represent our data as a point in a coordinate system.

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


# Graphical Intuition

Here we will have a Graphical Intuition of Machine Learning with an Example.

To get a graphical intuition we use all the concepts we discussed previously.

Here we take the example data which we discussed in our 1st class.


Tip amount prediction data.
In the above dataset, we have two columns 1) Bill amount 2) Tip amount.

we already discussed how to represent the data in coordinate system.

We have two columns so we represent the data in the two-dimensional coordinate system.
Now if anyone has given us a task.

Task: To identify a line that passes through the middle of the plotted data points.
From the knowledge obtained from our previous discussions.

We have to identify a line that passes from the middle of the data.

From our last discussion, we know that the equation of a line is given as y=mx+z

m = slope

z= intercept

The above equation we convert to y=w1x+w0

Just we changed the variable names.

From now we use this new terminology because most of the machine learning textbooks follow this terminology.

As we learned slope value ie w1 value gives slant and direction.

Intercept value ie w0 gives the value at which the line bisects the y coordinate

With this knowledge what we do is select randomly w1,w0 values.

Let's say the w1=10 and w0 = 0. these are the values that we selected.

The line with these values is shown in the graph above. This line is in red color.

As w0 =0 means the line is passing through the origin.

Now change  w1 value so that slantness increases.
The lines for w1=20,w1=30,w1=40 are shown in the diagram above.

But none of the lines matches to our required line.

Let's take w1=10 and w0 =15.

This line bisects the y coordinate at value 15.
Now change w1=20, w1=30,w1=40. All these lines are shown in the above graph.

But none of the lines matches our required line.

In this way, we change different w1 and w0 values until we find the line required.

This graphical representation of how lines are changing should have in your mind.

This helps you a lot in understanding machine learning algorithms.

Most of the machine learning algorithms identify w1 and w0 values that satisfy the given task.

Don't forget the task here. To identify a line that passes though the middle of the data.

In this way, w1 and w0 can be adjusted to find the line that passes through the middle of the data.

How this w1 and w0 will adjust?

We will learn this in our next discussions in Gradient descent.

This graphical intuition of machine learning will be extended to N-Dimensional coordinate space in the next discussion


# Understanding Plane and Hyperplane 



Here we will have an understanding plane and hyperplane for machine learning with an example.
In our last discussion, we had a Graphical Intuition of machine learning algorithms.

The example which we considered is giving intuition on a 2-dimensional coordinate space.

We will extend our concept from 2 dimensional to 3-dimensional coordinate space and to N-Dimensional coordinate space.

Before going into the concept first we have to understand the concept of

1)Point

2)Plane

In 3-dimensional coordinate space.

lets take a point (1,2,15) in 3 dimensional coordinate space.

In 3 dimensional coordinate space, we have x coordinate, y coordinate, z coordinate.

To plot the point in 3 dimensions move 1 unit on the x-axis and 2 units on the y-axis and 15 units on the z-axis.

Space which humans can visualize comes to 3-dimensional coordinate space.

Like we have lines in 2-dimensional coordinate space.

We have planes in 3-dimensional coordinate space.

Equation of a line is given as ax+by+c=0

We convert this equation to another form y = mx+c.

This conversion is done in our previous discussion.

Equation of a plane in 3 dimmensional space given as ax+by+cz+d=0

cz= - ax -by – d

z=    -  (a/c)x  - (b/c)y – d/c

a, b, c, d are constants.

Constant divided by constant is constant.

We give naming as z = w1x + w2y + w0.

By changing w1, w0 values in the equation of a line. the position of the line changes.

By changing the values of w1, w2, w0 the position of the plane changes in three-dimensional coordinate space.

Try to visualize this concept by understanding the last discussion made on a 2-dimensional coordinate space.
The above table has three columns. So we can represent this data in 3-dimensional coordinate space.

Task: identify a plane that passes through the middle of the data.

How we find the plane that passes through the middle of the data.

Modifying w1,w2,w0 we can change the position of the plane.

Most of the machine learning algorithms task is to identify the w1,w2,w0 values that fit our task.

Similarly, we have an N-dimensional coordinate system.

The equation of the plane in N-dimensional space is given as ax1+bx2 +...+nxn+ z=0

The plane above the 3-dimensional coordinate system we call it hyperplane.

We can write the equation of hyperplane n= w1x1+w2x2+....+wn-1x+w0
The above table has n columns so we represent the data in the N-dimensional coordinate system.

Task: the machine learning algorithm has to identify a plane that passes through the middle of the data.

By changing the w1, w2,.... wn-1, w0 values. We can change the position of the hyperplane until we reach our task.

with this understanding of the plane and hyperplane concept. we can learn about machine learning algorithms easily



# Derivative of a Function

Here we discuss the derivative of a function for machine learning.

Let's take an example and understand what derivative means.

Take a line in two-dimensional coordinate space.

Task: To identify the slope of the line.

We had this discussion previously on how to identify the slope of a line.

Take two points on the line. Lets say as (x1,y1) and (x2,y2)

The slope of the line is given as y2-y1/x2-x1.

The equation of a line is a linear function.

Linear function: the function that goes straight. It doesn't take any turns.

The slope of a linear function is constant. ie at any point on the line slope is the same.

Understand slope means slant. Will the slant of a line change? No

at any point on the line, the slant is the same.

Let's take a nonlinear function.

Eg: y = (x)^2

The function looks like in the below figure.

The slant of the function at the above two points are not the same because function taking turns.

Let's take two points on the function. In the above figure, it's given in red color.

Another point in green color. The slant changes from point to point shown in the line.

Task: identify the slope of the function at a given point.

Lets say at x = 5.

The slantness changes from point to point.

How to calculate the slope at the given point?

Follow the below procedure for finding the slope of a function at a given point.

Point is (x1,y1) =(5,25)

Now consider one more point on the plane which is infinitesimal close to the given point.

Eg: (x2,y2)= (5.000009, 25.009081) is a point which is very very close to (5,25)

let's say the change in y as ∆y.

change in x as ∆x.

∆x=x2-x1

∆y=y2-y1

From the above equations

y2= y1+∆y

x2=x1+∆x

From our function y=x^2

y1=(x1)^2---eq 3

y2=(x2)^2---eq 4

From the above equations substitute y2= y1+∆y in equation 4.

Similarly, substitute x2=x1+∆x in equation 4

we get ( y1+∆y)= (x1+∆x)^2----eq 5.

substitute eq 3 in eq 5.

we get x1^2 + ∆y = (x1+∆x)^2—eq 6.

expand eq 6.

we get x1^2 + ∆y =  x1^2 + 2x1∆x + ∆x^2----eq7.

∆y=2x1∆x+∆x^2----eq8.

Divide both sides by ∆x.

∆y/∆x=  2x1∆x+∆x^2/∆x.

Taking ∆x common on the numerator and cancel with denominator we get

∆y/∆x=2x1+ ∆x---eq 9

As we considered the change in x ie ∆x which is very low .so we neglect ∆x from eq 9.

∆y/∆x=2x1---eq10.

Substitute x1 in eq10 we get 2*5 = 10.

slope at(5,25) is 10.

As we remember from our calculus derivative of the function y= x^2 is dy/dx=2x.

That is what we got in eq 10.

So the derivative of a function is the slope at a given point.

Derivative: if f is a function than derivative of a function f we call it f' is a function which calculates

the slope of a function f at a given point.

We can solve derivative formulae using the above process.

You try on your own to find the slope of the function y= x^3

we know the derivative of y=x^3 is dy/dx= 3x^2.


# Finding the Maximum and Minimum of a Function

In this class, we will understand how to finding the maximum and minimum of a function for machine learning.

Take an example function y= (x-1)^2 +1.

Task: to identify minimum point on the function. Ie at what value of x y is minimum.

We observe from the above figure slope at the minimum point is zero.

At minimum point-slope is horizontal line ie its zero.

The above function doesn't have the maximum point.

Let's check why?

The right side of the minimum point as x increasing y increasing.

The left side of the minimum point also y keeps on increasing.

It will increase till infinity. So we don't have the maximum point.

The derivative of the function finds the slope of the function at a given point.

This we discussed in our previous discussion.

By using the concept of a slope at a minimum point is zero.

So dy/dx = 2x-2

equate it to zero.

2x-2=0

x=2/2

x=1.

At x=1 we have minimum y value.

Let's take another example

y=-(x)^2

The graph of the function is given in the figure below.

From the above figure, we can observe it has maximum point.

The above function doesn't have a minimum point.

At the maximum point. the slope is zero.

So derivate the equation and equate to zero.

Solve that equation for finding x.

In our next discussion, we use a method called gradient descent for finding the minimum point on a function.

Here we discuss the situation at which the gradient descent is helpful.

Lets take an equation y = cos(x)+tan(x)/e^x

we can derivate the above function.

We equate the derivative to zero.

But solving the x value from the equation is difficult.

In this type of situation, we use a gradient descent method.


# Understanding Gradient Descent 


In this class, we will have an understanding gradient descent for machine learning with an example.

This is a computational method for finding the minimum point on a function.

At what value of x we have minimum y.

The situation at which this method is useful is discussed in the previous discussion.


Let's take an example and understand how gradient descent works.

Take function y=(x-1)^2 +1.

Before going to concept lets refresh some of the concepts.

Derivative gives the slope of a function at a given point.

Slope means a change in y /change in x.

The slope is +ve if the increase in x at a given point y also increase

The slope is -ve if the increase in x at a given point y decrease

To identify the minimum point.

Randomly select an x value. Here we are selecting x=5

Derivative of the function is dy/dx=2x-2.

substitute 5 in the derivative equation ie 2*5 -2 =8.

so slope at x =5  is 8

+ve slope means x increase y increase.

Understand we have to identify the minimum y value. Ie reduce x value.

Lets take x= -5.

At x=-5 slope = 2*-5 -2 = -12.

The slope is -ve means x increase y decrease.

Y moving towards minimum. So increase x.

To meet the above conditions gradient descent uses the equation xnew = xold - alpha*[dy/dx] xold.

Check the above equation if the slope is +ve we subtracting from xold value ie decreasing x value.

If the slope is -ve than -ve * -ve we get +ve means we are adding value to xold. Increasing x value.

The above equation always push x value to the minimum y value.

Assume alpha = 0.2. we will understand why we use alpha at the end of the discussion.

We understand this wit an example

let's take x=5

so xold =5

find xnew value

xnew = xold – alpha*[dy/dx]xold  here xold = 5 dy/dx = 2x-2.

xnew = 5 – 0.2 * [2*5-2].

xnew = 5 – 0.2 * 8.

xnew = 5 – 1.6.

xnew = 3.4.

Observe from the above figure x moving to minimum y point.

Now xold = 3.4

again find xnew

xnew = xold – alpha [dy/dx]3.4

xnew = 3.4 – 0.2 * [(2 * 3.4) – 2]

xnew = 3.4 – 0.2 * 4.8

xnew = 3.4 – 0.96

snew = 2.44

again x moving near to minimum y value.

So keep repeating this computation till xnew value reaches minimum y value.

How we know x reaches to minimum y value.

Observe from the above figure when x reaches to point p1. When we find the xnew value at p1.

Xnew moves to p2 point. P2 is on the other side.

At p2 slope is -ve. So when we find xnew again at p2 we increase x value and move to p1 side again.

So x value swaps between xnew -xold value.change in value is very very small. And x value is not changing much.

This we call it convergence. Ie x moved to minimum y. we can stop computing.

Let's understand what's the use of alpha.

Lets take alpha = 0.4.

xnew = xold – alpha*[dy/dx]xold  here xold = 5 dy/dx = 2x-2.

xnew = 5 – 0.4 * [2*5-2].

xnew = 5 – 0.4 * 8.

xnew = 5 – 3.2.

xnew = 1.8.

Observe as the alpha value increased fro 0.2 to 0.4.

x value takes a long jump.

At alpha = .2 x jumped from 5 to 3.4.

at alpha = .4 x jumped from 5 to 1.8.

as alpha value increases x takes long jump ie x moves to minimum point very fast.

Convergence is fast with a big alpha.

But with large alpha, the problem is we can not have a better approximation.

Why not better approximation?

Observe from the above figure.

The point in red color. From there, its taking long jump means its moving to the other side.

Observe the x value swaps on both sides. And it's far from the actual minimum.

That's the reason large alpha doesn't have better approximations.

If needed better approximation use small alpha.

if needed fast convergence use large alpha.

This understanding of gradient descent will help a lot in Machine learning.


# Mathematics behind Linear Regression 

In this class we will discuss about Mathematics behind Linear Regression

We take an example and understand Mathematics behind Linear Regression.

It s having bill amount and tip amount.

The data is having two columns. So we can represent the data in two-dimensional coordinate space.

From our first discussion about machine learning.

Machine learning is about

1) collecting the data

2) identifying the mathematical formulation from the data

3) use this mathematical formula for predicting the future values

For simplicity, we collected 4 data points.

In reality, we will have thousands of data points.

The mathematical formulation which we do in linear regression is to identify a line that passes from the data.

Lets say y = 0.1 x + 2.

To predict future tip amount.

If a new customer came into the restaurant and he made a bill amount of 500.

we know bill amount here. So we know the x value

substitute x value in the equation y= 0.1*500 +2.

y= 52.

This is how we predict the tip amount using the mathematical formulation.

But which line we have to choose.

We have to choose the best line that had the minimum loss.

Let's understand the minimum loss by taking an example.

Take 2 lines yz1 = 0.1x+2.

yz2=0.108x+1.

Consider the predicted data using the above two lines.

The Tip amount column shows the actual tip amount given by the customer.

yz1 and yz2 columns predicted by lines yz1 and yz2 respectively.

actual value – predicted value = loss.

Compute loss to all the points.

The loss for two lines given.

square the loss to avoid -ve value.

Why we have to square is understood at the end of the discussion.

After squaring add all the values.

Total loss is 148 for yz1 and a184 for yz2

so yz1 is having minimum total loss.

The linear regression model is to identify a line that having a minimum loss.

Mathematically it is given as

argmin ∑ (y - yz)^2

y actual value

yz predicted value

The above mathematical argmin means find the minimum of ∑(y-yz)^2.

These types of problems we call it as optimization problems.

These types of optimization problems can be solved using Gradient descent.

Introduction to linear regression gives a better understanding of machine learning.
