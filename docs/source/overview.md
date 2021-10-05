# Overview

Insolver is a low-code machine learning library, initially created for the insurance industry, but can be used in any other industry as well. 

Data Science and Machine Learning are very popular nowadays. More and more industries are starting to use models in their business processes. However, it is not easy to adapt general algorithms to specific situations. For many small companies, specific specialists can be very expensive. Our project Insolver is a low-code solution that simplifies building machine learning models focused primarily on insurance industry. 

The solution also provides models for interpretation, not a "black-box". The insurance market is highly competitive and the quality of the models used (especially models of risk assessment and client scoring) affects the quality of the portfolio and the profitability of the company. In recent years, open-source packages have appeared that allow building models without special software. Extensive experience in the insurance business allows us to create high-quality machine learning models that take into account the features and peculiarities of insurance companies. The approach is aimed at creating a versatile solution that can be used in both large and small companies. 


Insolver library is designed to automate the way how insurance company does their calculation. In 2018, we, Insover team, decided to create a product that will help insurance companies solve certain problems of calculating and deploying a model. As analysts and participants of different projects, we wondered we wondered what are the most common challenges insurance companies face with their machine learning models and operations? As a result, we created the free open-source library Insolver.


Insolver python library should help insurance and other industries companies to solve a very simple in words and a difficult task in practice - to prepare data, build and deploy machine learning models.

## Key parts

In our solution, we implemented such things like data transformations that work in production, model validation methods for a business need, ensuring Gradient boosting model quality using shap values, comparing of GLM and boosting models, and other things.

Insolver works as an additional wrapper for modern open-source libraries such as sklearn, h2o and others. 

In general, you can create your own wrapper interface for any library you like using insolver template. This can be quite powerful in terms of industry-specific tasks and significantly reduce the time to market for ML models.

We welcome specialists from other industries to improve insolver for their industrial tasks.

Many small and medium-sized insurance companies often require standard solutions for pricing systems and risk assessment systems. Accordingly, Insolver can be such an open-source solution.


In addition to typical business problems, insurers face with fairly common problems related to dirty data, incomprehensibly filled fields, failures resulting from failures in systems. Incomplete description of data and fields in their data. We understand these issues and offer techniques to fill missing data with saving distributions.

## https://mset.space cloud
One of the problems analysts in insurance companies face is large amount of data and, consequently, lack of resources to process it. Companies address these problems in a variety of ways. We also want to offer a solution related to cloud computing resources https://mset.space. Insolver is naturally implemented in the mset cloud servers.

## Deployment
Flask and Fast API servers implementation with Insolver allows using such product during production.

## Model quality control
Also there is a problem of model degradation over time and Insolver offers a solution to this problem 

To solve such challenges we offer analysts the Insolver product, which is designed to solve these problems.

### Key parts of the Insolver project:

1. The idea of an updated pandas dataframe as InsolverDataFrame with additional functionality.
2. Deployable transformations, which allow to create and register data transformation before the model goes into production.
3. Simplified low-code interface for model train and development
4. Models comparison part
5. Models quality estimation via shap values
6. Model deployment as a microservice

**Insolver is open source, which allows you to use it anywhere you want.**

We also provide a proprietary enterprise Insover solution  based on the Insolver project to solve highly specialized tasks that can be built for a specific customer and includes, among other things, the implementation of modern state-of-the-art algorithms. Now there is no need to spend time implementing these technologies on your own.

Overall, Insolver library is open source library that automates calculation and implementation of ML models, primarily in the insurance industry. You can use Insolver on your servers or use cloud https://mset.space where Insolver is installed and can be used naturally.


