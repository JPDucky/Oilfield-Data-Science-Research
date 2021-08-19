# Time-Series-Methods-Research
A comparison of Classical Analytical Techniques versus Modern Data Science Techniques in the forecasting of oil well production over the lifetime of the well. 

In this paper I detail various methods of analyzing oil well production using a combination of both machine learning algorithms and modern analysis techniques. My findings show that using these techniques produces results over an order of magnitude greater than classiscal analysis techniques (2.5%-5% vs. 20%-30% accuracy).

I intend on increasing this research to include the domain of multiple enhanced oil recovery recovery units over an entire field in the multivariate analysis section, with an attempt to expand into the domain of predicting new wells drilled.

---

ABSTRACT:

I have attempted to analyze multiple data analysis techniques with machine learning and compare them to more traditional methods of analysis, and in doing so find that they are much more robust and accurate than the traditional models.

INTRODUCTION:

Classical Analysis of historical data from oil wells for the prediction of future forecasting has commonly used techniques such as Monte Carlo Analysis, Production Decline Curves, or Mass Balance Equations for the modeling of predicted well output. While these models give a decent enough analysis for economic use of the predictions they output, they often involve a fair bit of guess work and accompanying uncertainty. My use of Data Analytics and Machine Learning attempts to seek out which variables available to the engineer are superfluous, and at the same time use machine learning algorithms to attempt to more accurately model production decline than previously available methods. 

Monte Carlo Analysis is a method of statistical risk analysis that determines the probability of an events occurrence by means of a large number of simulations that attempt to model all possible situations of a set of given parameters. The drawbacks of this method is that it both may take into account unnecessary variables, making the model both inefficient and possible inaccurate, and it may also fail to take into account necessary variables. It assumes “All distributions are normal and correlations are zero, it does not accurately capture the interelationships [sic] between multiple variables contained in historical data, therefore, it does not depict real world complexities” (Robinson, Keith). This method is therefore not only inappropriate for modeling,  but it may even introduce uncertainty into the model instead of reducing it. 

Decline Curve Analysis is often used to predict the decline in production over the lifetime of an oil well. It is a graphical method of analysis that uses line fitting to attempt to model the production of a well, but typically it needs 3 to 5 years of stabilized production data before it is usable, but it still has a great deal of uncertainty as it relies on the estimation of parameters and model functions. This, too, is unacceptable in terms of accuracy, and a more accurate method of analysis must be found.

  The Mass Balance equation uses a Runge-Kutta technique to estimate the rate at which the well will decline by means of a derivate equation, but does not take into account the previous histories of the well’s production and the effects the trend will have on the forecast. It is in this area that I believe a solution exists within the realm of machine learning.

  I believe that the problems of uncertainty, the failure of current models to take into account immediately preceding historical data beyond one time step, and the elimination of unnecessary variables can be solved with the use of Principal Component Analysis, Vector Autoregression, and LSTM algorithms. In this report I will be the analyzing the Red River B enhanced oil recovery unit in the Buffalo River Field in Harding county, South Dakota. With the use of the above algorithms, I intend on finding and eliminating unnecessary variables with the use of Principal Component Analysis and Vector Autoregression, and then I will attempt to model the future output variables with use of an RNN known as LSTM (Long Short-Term Memory). This algorithm will attempt to address the inadequacy of current models by taking into account historical variables with a time lag that will look back over a set interval and use those variables influence on the proceeding variables to predict future variables, and give us a more accurate prediction than other currently available models.

DESCRIPTION:

This project involves the following datasets:

    • 1 excel .csv file containing the combined and cleaned data from the other sources
    • 2 excel files from the South Dakota Department of Natural Resources
    • A paper detailing the reservoir parameters by Sippel, Luff, Hendricks, and Eby


  The injection and production data from the South Dakota Department of Natural Resources came in two separate files, one for injection and one for production. The paper contains reservoir data on reservoir parameters and the quality of those parameters, as well as the chosen field’s candidacy for water injection. The oilfield data dates back to May of 1987, and an update of parameters is given monthly with an average value of the measured variables over the course of the month for each time slot. The injection and production files contained data for every well in the state, which was also grouped by field and enhanced oil recovery unit (EORU). I chose to analyze the data summed over the Red River B EORU in the Buffalo River field in Harding County, South Dakota.
 
 Initially I combined the data for the field from the two injection and production files into one excel file, then I exported it as a .CSV file so that I may use it for analysis with python. The variables I chose to use were: Date, Summed Total Injected Volume, Days Injected, Summed Injection Rate- Daily Average, Sum Injection Pressure- Daily Average, Summed Max Pressure, Oil Produced over the Month, Gas Produced over the Month, Water Produced over the Month, Water Cut, Cumulative Water Injected into the EORU, Cumulative Oil Produced from the EORU, Cumuative Water Produced from the EORU, Cumulative Gas Produced from the EORU, Average Thickness of the Reservoir, Porosity, Permeability, Compressibility, Saturation of Water, Oil Denisty, Abandonemnt Pressure, Viscosity at Bubble Point, Formation Volume Factor, and the Initial Reservoir Pressure. With Excel, I summed the oil, water, and gas production and water injections to get a cumulative total over the Unit with respect to time. My code comes from Dr. Jason Brownlee’s Website “Mastering Machine Learning”, and I used his code for Principal Component Analysis and the LSTM, and I got the code for the Vector Autoregression form Statsmodels.com, a free online learning resource. I ran the code with the Spyder IDE, and used the scikit-learn, numpy, matplotlib, pandas, keras, tensorflow, and seaborn packages within it.

METHODS:

  Using the attached codes available, I performed and exploratory data analysis on the dataset after converting the excel file to a .CSV file, and I created a distribution map and heatmap using the scikit learn statistical analysis code to find and identify which variables could be eliminated, so that I may only use necessary inputs for the proceeding algorithms. My hope was to use Principal Component analysis for dimensional reduction, but I could not get it working for a multivariate application. I then used a Vector Autoregression algorithm to get a correlation matrix for the 5 variables I chose to analyze, Cumulative Oil, Gas, and Water Production and Injection, and Summed Max Pressure, but I cannot yet figure out how to get my code to analyze more variables than that. It was my intention of using this to find an equation to predict other variables, but it I cannot get that out of my code yet either. I had hoped to use this equation in correlation with the predictions from a univariate LSTM to try and predict and forecast other variables, but as I could not get the Vector Autoregression working I decided to try and use a multivariate LSTM and predict all of the variables at once, so I edited the code I used from Dr. Jason Brownlee’s website into a multivariate LSTM from a univariate LSTM, and once I did some tests with that algorithm, I decided that I needed an even more accurate prediction, so I edited the code to account for a moving time lag window, and then I tested each available loss function with every optimizer and settled on the mean square error and the “NADAM” optimizer. I arrived at this combination by testing for RMSE and settled on the lowest score with the least variation. The resulting plot of train and test scores was skewed due to the lookback window, so I had to create a dummy matrix of zeros to pad the beginning of the prediction.

RESULTS:
	
  My results showed that the LSTM neural networks is able to accurately predict the production with an extremely small error and to a much more accurate degree than with conventional methods. I am still unable to get my code to predict future forecasts, which involve feeding the model its own previous predictions and using those to further predict new variables out to a specified time following the pattern it learned during training. I am confident enough of the model’s accuracy to predict future variables to say that it would be useful in an economic prediction.

  With my exploratory data analysis, I found that the most well correlated variables were the cumulative injection and productions, so I decided to analyze the 4 that best correlated for use in the LSTM.
  
![image](https://user-images.githubusercontent.com/34105363/129991343-c0882f93-e224-43c9-81e4-95ab263e94cb.png)


Correlation Plot of Variables

![image](https://user-images.githubusercontent.com/34105363/129983467-f84c04a9-e130-4e4f-9841-10135cadfb08.png)


Heatmap of Variables

![image](https://user-images.githubusercontent.com/34105363/129983509-233bcf8b-2ff9-4c30-bfad-456e4a0a7411.png)



I performed the regression on these variables and was able to get their statistical values. 

Regression Analysis of 4 Best Variables

![image](https://user-images.githubusercontent.com/34105363/129983524-1f01b889-da3c-430f-bd3f-3e2b90a35078.png)



I then used the univariate LSTM to analyze Oil Production and was able to get a decently accurate prediction. 

![image](https://user-images.githubusercontent.com/34105363/129983635-dd2f3c35-e3b6-4dfd-af92-07b689aee8a4.png)



I have attempted multivariate analysis on stock data, and can get them to predict an accurate forecast, but I cannot get the code to finish running on my production data to plot the actual forecast. I will include both files.


CONCLUSIONS:

  I was able to demonstrate the effectiveness of an LSTM and exploratory data analysis on summed field production and injection data, and demonstrate it’s reliability and effectiveness to be more accurate than previous methods. This project has shown me that machine learning analysis is extremely powerful and is a very effective method for analyzing and predicting data I will continue to work on in an attempt to succeed on my original plan and will update this report as I collect my findings. 



----
References

Dr. Mayank Tyagi, for help and guidance in coming to a conclusion

Dr. Brownlee, Jason. “Machine Learning Mastery.” www.Machinelearningmastery.com

Sippel, Luff, Hendricks, Eby. “RESERVOIR CHARACTERIZATION OF THE ORDOVICIAN RED RIVER FORMATION IN SOUTHWEST WILLISTON BASIN BOWMAN CO., ND AND HARDING CO., SD”. Web. 

South Dakota Department of Natural Resources. http://cf.sddenr.net/sdoil/index.cfm?index=New+Search. Web.

StatsModels Statistics in Python. “Vector Autoregression.” Web.

Robinson, Keith, Jr. “The Problem with Monte Carlo Simulation.” AVANCED ANALYTICAL TECHNIQUES. Web. http://advat.blogspot.com/2017/11/the-problems-with-monte-carlo-simulation.html
