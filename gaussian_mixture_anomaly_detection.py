#!/usr/bin/env python
"""
Copyright (c) 2017 Sergei Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Partially base on http://www.sciencedirect.com/science/article/pii/S0968090X16000188
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy


class GaussianMixtureInTimeAnomalyDetector:
    '''
        ClusterAD-DataSample anomaly-detection method implementation
    '''
    def __init__(self,
                    n_components=35,
                    tol=1e-6,
                    covariance_type='diag',
                    init_params='kmeans',
                    max_iter=100,
                    random_state=None,
                ):
        '''
            Constructor accepts some args for sklearn.mixture.GaussianMixture inside.
            Default params are choosen as the most appropriate for flight-anomaly-detection problem
            according the original article.
        '''

        self.n_components = n_components
        self.tol = tol
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.random_state = random_state
        self.max_iter = max_iter

        self.eps = 1e-12  # feature-normalization constant

    def fit(self, X):
        '''
            X must contains F objects time series with length T vectors-features with size N each
            i. e. X.shape is (F, N, M)
        '''
        from sklearn.mixture import GaussianMixture
        from numpy.linalg import norm
        from copy import deepcopy
        
        print('Run fitting')

        X = np.array(X)

        assert len(X.shape) == 3
        self.F, self.T, self.N = X.shape

        # prepare data for fitting
        X = X.reshape(self.F * self.T, self.N)

        self.data_mean = np.mean(X, axis=0)
        self.data_std = np.std(X, axis=0) + self.eps

        X = self._normalize(X)

        gm = GaussianMixture(
            n_components=self.n_components,
            tol=self.tol,
            covariance_type=self.covariance_type,
            init_params=self.init_params,
            random_state=self.random_state,
            max_iter=self.max_iter,
                            )

        gm.fit(X)

        self.X = X.reshape(self.F, self.T, self.N)

        self.cluster_weights = gm.weights_
        self.cluster_means = gm.means_
        self.cluster_covariances = gm.covariances_

        print('Start probabilities memorization')

        self.__memorize_probs()

        return self

    def predict(self, X, times):
        print('Normalization')
        
        X = np.array(X)
        X = self._normalize(X)
        
        print('Start prediction calc')
        
        return np.array([self.__evaluate_sample_in_time(X[i], times[i]) 
                           for i in tqdm(np.arange(X.shape[0]),position=0)])

        


    def __memorize_probs(self):
        # memorization all P(cluster|sample)
        self.__p_cluster_sample = np.zeros((self.n_components, self.T, self.F))

        for series in tqdm(np.arange(self.F), position=0):
            for time in np.arange(self.T):
                probs = [self.cluster_weights[i] * self.__p_sample_cluster(self.X[series][time], i) \
                    for i in np.arange(self.n_components)]
                norma = np.sum(probs)

                for cluster in np.arange(self.n_components):
                    self.__p_cluster_sample[cluster][time][series] = probs[cluster] / (norma + 1e-9)

        # memorization all P(cluster|time)
        self.__p_cluster_time = np.zeros((self.n_components, self.T))

        for time in np.arange(self.T):
            for cluster in np.arange(self.n_components):
                self.__p_cluster_time[cluster][time] = self.__get_p_cluster_time(cluster, time)


    def _normalize(self, X):
        return (X - self.data_mean) / self.data_std

    def __diag_gauss_pdf(self, x, mean, cov):
        '''
            Custom calculation gaussian density in case if covariance is diagonal matrix

            cov is array of covariance matrix diagonal elements
        '''
        delta = np.array(x) - np.array(mean)
        inv = 1 / (np.array(cov) + self.eps)
        logp = -0.5 * ((delta.dot(inv * delta)) + np.log(np.prod(cov) + self.eps) + self.N * np.log(2 * np.pi))

        return np.exp(logp)

    def __p_sample_cluster(self, x, cluster):
        '''
            Conditional likelihood(sample|cluster)
        '''
        return self.__diag_gauss_pdf(x,
                                    self.cluster_means[cluster],
                                    self.cluster_covariances[cluster])


    def __get_p_cluster_time(self, cluster, t):
        clusters_probs = [np.sum(self.__p_cluster_sample[i][t]) for i in np.arange(self.n_components)]

        return clusters_probs[cluster] / np.sum(clusters_probs)

    def __evaluate_sample_in_time(self, x, t):
        '''
            Evaluation in population log likelihood for time-slice normalized sample x_t.

            x is M-dimentional one-time-slice sample

            t is order(time) of sample x in time series
            t must be in [0, 1, ... N)
        '''

        return np.log(1e-200 + np.sum([self.__p_sample_cluster(x, cluster) * self.__p_cluster_time[cluster][t] \
                         for cluster in np.arange(self.n_components)]))


    def __evaluate_log_likelihood(self, X):
        log_likelihood = np.zeros(X.shape[:2])
        for f in np.arange(X.shape[0]):
            for t in np.arange(X.shape[1]):
                log_likelihood[f][t] = self.__evaluate_sample_in_time(X[f][t], t)

        return log_likelihood
 