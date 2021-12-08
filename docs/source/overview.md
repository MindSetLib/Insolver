# Overview

Insolver is a low-code machine learning package initially created for the insurance industry but can be used in any other industry. 

Data Science and Machine Learning are very popular nowadays. More and more industries are starting to use models in their business processes. However, it is not easy to adapt general algorithms to specific situations. For many small companies, specific specialists can be costly. Our project Insolver is a low-code solution that simplifies building machine learning models focused primarily on the insurance industry. 

The insurance market is highly competitive, and the quality of the models used (especially models of risk assessment and client scoring) affects the quality of the portfolio and the company's profitability. In recent years, open-source packages have appeared that allow building models without special software. Extensive experience in the insurance business will enable us to create high-quality machine learning models that take into account the features and peculiarities of insurance companies. Our solution aims to create a versatile solution that can be used in large and small companies. The package also provides models for interpretation, not a "black-box." 


In 2018, the Mindset team decided to create a product that will help insurance companies solve specific problems of calculating and deploying a model. As analysts and participants of different projects, we wondered what the most common challenges insurance companies face with their machine learning models and operations are? As a result, we created the free, open-source package Insolver. The insolver package is designed to automate how insurance companies do their calculation.


Insolver python package should help insurance and other industries companies to solve a very simple in words and a difficult task in practice - to prepare data, build and deploy machine learning models.

## Key parts

In our solution, we implemented data transformations in production, model validation methods for a business need, ensuring Gradient boosting model quality using SHAP values, comparing GLM and boosting models, and other things.

Insolver works as an additional wrapper for modern open-source libraries such as sklearn, h2o, etc. 

In general, you can create your wrapper interface for any package you like using the insolver template. It can be pretty powerful in industry-specific tasks and significantly reduce the time to market for ML models.

We welcome specialists from other industries to improve insolver for their industrial tasks.

Many small and medium-sized insurance companies often require standard solutions for pricing systems and risk assessment systems. Accordingly, Insolver can be such an open-source solution.


In addition to typical business problems, insurers face fairly common issues related to dirty data, incomprehensibly filled fields, failures resulting from failures in systems. Also, incomplete description of data and values in their data. We understand these issues and offer techniques to impute missing data with saving distributions.

## https://mset.space cloud
One of the problems analysts in insurance companies face is a large amount of data and, consequently, the lack of resources to process it. Companies address these problems in a variety of ways. Insolver is naturally implemented in the mset cloud servers. We also want to offer a solution related to cloud computing resources https://mset.space.

## Deployment
Flask and Fast API servers implementation with Insolver allows using such services during production.

## Model quality control
Also, there is a problem of model degradation over time, and Insolver offers a solution to this problem. 

We offer analysts the Insolver product to solve such challenges, which is designed to solve these problems.

### Key parts of the Insolver project:

1. The idea of an updated pandas DataFrame as InsolverDataFrame with additional functionality.
2. Deployable transformations allow creating and registering data transformation before the model goes into production.
3. Simplified low-code interface for model train and development
4. Models comparison part
5. Models quality estimation via SHAP values
6. Model deployment as a microservice

**Insolver is open source, which allows you to use it anywhere you want.**

We also provide a proprietary enterprise Insolver solution based on the Insolver project to solve highly specialized tasks that can be built for a specific customer and includes, among other things, the implementation of modern state-of-the-art algorithms. There is no need to spend time implementing these technologies on your own.

Overall, the Insolver package is an open-source library that automates the calculation and implementation of ML models, primarily in the insurance industry. You can use Insolver on your servers or use cloud https://mset.space where Insolver is installed and can be used naturally.
