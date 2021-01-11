# Overview

Insolver is low-code machine learning library, initally created for insurance industry, but can be used in any other.

Data Science and Machine Learning are very popular nowadays. More and more industries begin to use models in their business processes. But an adaptation of general algorithms for exact situations is not a simple deal. For many small companies, specific specialists could be very expensive. Our project Insolver is a low-code solution that simplifies building machine learning models focused primarily on the use of insurance companies. 

The solution also provides models to be interpreted, not the ‘black-box’. The insurance market is highly competitive and the quality of the models used (especially models of risk assessment and client scoring) affects the quality of the portfolio and the company's profitability. In recent years open-source packages have appeared that allow building models without specialized software. Our great experience in the insurance business allows us to create high-quality machine learning models that take into account the features and peculiarities of insurance companies. The approach is aimed at creating a fairly universal solution that can be used in both large and small companies. 


Insolver library aims to automate the way how insurance company does their calculation. In 2018 we as an Insover team decided to create a product that helps insurance companies to solve certain model calculation and deployment problems. As analysts and participants of different projects, we thought about which problems are insurance companies meet very often with their models and machine learning operations? As a result, we created Insolver free open-source library.


Insolver python library should help insurance and other industries companies to solve very simply in words and difficult in practice task - prepare data, build and deploy machine learning models.

# Key parts

In our solution, we implemented such things as data transformations that work in production, model validation techniques as business need, Gradient boosting model quality assurance via shap vales, comparison of GLM and boosting models, and other things.

Insolver works as an additional wrapper for modern open-source libraries such as sklearn, h2o and others. 

In general, you can create you own wrapper interface for any library you like using insolver template. This can be quite powerful in terms of industry-specific tasks and significantly decrease time to market for ML models.

We are welcome specialists from other industries to improve insolver for their industrial tasks.

Many small-sized insurance companies and medium-sized insurance companies often require standard solutions for pricing systems and risk assessment systems. And Insolver can be such open-source solution.


In addition to typical business problems, insurers are faced with fairly typical problems related to dirty data, incomprehensibly filled fields, failures resulting from failures in systems. Incomplete description of data and fields in their data. We understand such problems and provide techniques to fill NA data with saving distributions.

# mset.space cloud
One of the problems faced by analysts in insurance companies is the problem of too much data and a lack of resources to process it. Companies are struggling with these problems using various methods. We also want to offer a solution related to resources for computing via cloud mset.space. Insolver is naturally implemented in the mset cloud servers.

# Deployment
Flask and Fast API servers implementation with Insolver allows using such product during production.

# Model quality control
Over time, the use of the model poses the problem of degradation of the irrelevant services of model models, which requires updating and operational implementation in the usage circuit, and these tasks are close to the concept of automatic machine learning autoML; other companies that also study this type of insurers more often have to deal with.


To solve such challenges we offer analysts the product Insolver, which is designed for these problems.

Basic part - Insolver community open source library.

## Insolver community key parts:

1. The first part of insolver community is the idea of an updated pandas dataframe as InsolverDataFrame with additional functionality.
2. Deployable transformations that allow to create and register data transformation before the model start to work in production.
3. Simplified low-code interface for model train and development
4. Models comparison part
5. Models quality estimation via shap values
6. Model deployment as a microservice

**Insolver is open source, which allows you to use it anywhere you want.**

We also provide Insover enterprise solution the proprietary part based on Insolver community and allowed you to solve highly specialized tasks that can be built for a specific customer and includes, among other things, the implement modern state-of-the-art algorithms. Now it is unnecessary to spend time implementing these technologies on your own.

Overall, Insolver library is open source library that automates calculation and implementation of ML models, first of all in the insurance industry. You can use Insolver community on your servers or use cloud mset.space where Insolver installed and can be used naturally.


