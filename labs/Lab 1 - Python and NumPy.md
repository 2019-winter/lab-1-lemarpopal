---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
Lemar Popal


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


1. What is the difference between JupyterHub (what we're using) and Jupyter Notebook?
2. How do we restart the kernel so that we can run all our code again?
3. How do we move notebooks between our local computer and the website?


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1


Problem A.1 Make an array a of size 6 × 4 where every element is a 2.

```python
import numpy as np
```

```python
a = np.full((6,4),2)
a
```

## Exercise 2


Problem A.2 Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1
everywhere else. (You can do this without loops.)

```python
b = np.ones((6,4))
np.fill_diagonal(b,3)
b
```

## Exercise 3


Problem A.3 Can you multiply these two matrices together? Why does a * b work, but
not dot(a,b)?

```python
a*b
```

```python
np.dot(a,b)
```

a*b multiples the matrices element-wise. 
np.dot(a,b) computes the dot product of two matrices, but it doesn't work. Because the arrays are 2-D, this is equivalent to matrix multiplication. The inner dimensions of a and b don't match so we can't matrix multiply them. 


## Exercise 4


Problem A.4 Compute dot(a.transpose(),b) and dot(a,b.transpose()). Why are
the results different shapes?

```python
np.dot(a.transpose(),b)
```

```python
np.dot(a,b.transpose())
```

For np.dot(a.transpose(),b):
The shape of a.transpose() is 4 x 6. The shape of b is 6 x 4. Therefore the shape of the result will be the outer dimension of the two matrices: 4 x 4. 

For np.dot(a,b.transpose()):
The shape of a.transpose() is 6 x 4. The shape of b is 4 x 6. Therefore the shape of the result will be the outer dimension of the two matrices: 6 x 6. 


## Exercise 5


Problem A.5 Write a function that prints some output on the screen and make sure you
can run it in the programming environment that you are using.

```python
def foo():
    print("bar")
```

```python
foo()
```

## Exercise 6


Problem A.6 Now write one that makes some random arrays and prints out their sums,
the mean value, etc.

```python
def random_arrays():
    # picks a random integer between 3 and 8
    rows = np.random.randint(3,8)
    cols = np.random.randint(3,8)
    
    # create a rows x cols matrix with random integers between 1 and 10
    m = np.random.randint(low = 1, high = 10, size = (rows,cols))
    
    print("Matrix:", m)
    print("Sum:", m.sum())
    print("Mean:", m.mean())
    print("m.sum(axis=0) (sum the columns)", m.sum(axis=0))
    print("m.sum(axis=1) (sum the rows)", m.sum(axis=1))
    
```

```python
random_arrays()
```

## Exercise 7


Problem A.7 Write a function that consists of a set of loops that run through an array
and count the number of ones in it. Do the same thing using the where() function
(use info(where) to find out how to use it).

```python
# create a 5 x 5 matrix with random integers between 1 and 10
array = np.random.randint(low = 1, high = 10, size = (5,5))
array
```

```python
def count_ones(array):
    count = 0
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[row][col] == 1:
                count += 1
    return count
```

```python
count_ones(array)
```

```python
###### ask 
# np.where() creates a boolean mask where array
def count_ones_where(array):   
    print(np.where(array == 1, 1, 0).sum())
```

```python
count_ones_where(array)
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.


Problem A.1 Make an array a of size 6 × 4 where every element is a 2.

```python
import pandas as pd
```

```python
a = pd.DataFrame(2, index = range(6), columns = range(4))
```

```python
# Or similarly using the code snippet from Exercise 1...
# pd.DataFrame(np.full((6,4),2))
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.


Problem A.2 Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1
everywhere else. (You can do this without loops.)

```python
###### Ask
# create 6 x 4 DataFrame filled with 1's
b = pd.DataFrame(1, index = range(6), columns = range(4))

# df.values returns a Numpy representation of the DataFrame.
# Then diagonal is filled with 3's.
np.fill_diagonal(b.values,3)
b
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.


Problem A.3 Can you multiply these two matrices together? Why does a * b work, but
not dot(a,b)?

```python
a.multiply(b)
```

```python
a.dot(b)
```

For the same reasons as Exercise 3:
a.multiply(b) multiples the data frames element-wise. 
a.dot(b) computes the dot product of two data frames, but it doesn't work. Because the arrays are 2-D, this is equivalent to matrix multiplication. The inner dimensions of a and b don't match so we can't matrix multiply them. 


## Exercise 11
Repeat exercise A.7 using a dataframe.


Problem A.7 Write a function that consists of a set of loops that run through an array
and count the number of ones in it. Do the same thing using the where() function
(use info(where) to find out how to use it).

```python
# creates a random Numpy array that is converted to a Data Frame
df = pd.DataFrame(np.random.randint(low = 1, high = 10, size = (5,5)))
df
```

```python
def count_ones_df(df):
    count = 0
    for index, row in df.iterrows():
        for i in range(len(row)):
            if row[i] == 1:
                count += 1
    return count
```

```python
count_ones_df(df)
```

```python
# same as Exercise 7 but input is a data frame, not a Numpy array
def count_ones_df_where(df):
    return np.where(df == 1, 1, 0).sum()
```

```python
count_ones_df_where(df)
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df["name"]
```

```python
# or similarly...
# titanic_df.name
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
# titanic_df.set_index('sex',inplace=True)
```

```python
# Select all passengers that are female:
titanic_df.loc["female"]
```

```python
# How many female passengers are there? 466
len(titanic_df.loc["female"])
```

## Exercise 14
How do you reset the index?

```python
titanic_df = titanic_df.reset_index()
```

```python

```
