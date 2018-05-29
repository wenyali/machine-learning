#!/usr/bin/env python
# _*_coding:utf-8 _*_

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import  datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/24"

iris_data = datasets.load_iris()
print("源数据：",iris_data.data[:3])

# svc 剖析
mod = LinearSVC(C=0.01,penalty="l1",dual=False).fit(iris_data.data,iris_data.target)
selectmod = SelectFromModel(mod,prefit=True)
data_svc = selectmod.transform(iris_data.data)
print("svc 剖析后的数据：",data_svc[:3])

# lasso 剖析
lassomodel = LassoCV()
selectmod1 = SelectFromModel(lassomodel,threshold=0.1).fit(iris_data.data,iris_data.target)
data_lasso = selectmod1.transform(iris_data.data)
print("lasso 剖析后的数据：",data_lasso[:3])


# lr 剖析
ir_model = LogisticRegressionCV(penalty='l1',solver='liblinear')
selectmod2 = SelectFromModel(ir_model,threshold=10)
selectmod2.fit(iris_data.data,iris_data.target)
data_lr = selectmod2.transform(iris_data.data)
print("lr 剖析后的数据：",data_lr[:3])


# 使用 l1 和 l2 综合选取特征
class LR(LogisticRegression):
    def __init__(self,threshold = 0.01,dual=False,tol=1e-4,C=1.0,fit_intercept = True,intercept_scaling = 1
                 ,class_weight=None,random_state = None,solver = 'liblinear',max_iter=100,multi_class = 'ovr',
                 verbose = 0,warm_start = False,n_jobs = 1):
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state, solver=solver, max_iter=max_iter,
                                    multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建 L2 逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练 L1 逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练 L2 逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1 逻辑回归的权值系数不为 0
                if coef != 0:
                    idx = [j]
                    # 对应在 L2 逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在 L2 逻辑回归中，权值系数之差小于设定的阈值，且在 L1 中对应的权值为 0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
def main():
    lr = LR(threshold=0.5,C=0.1)
    sub_mod = SelectFromModel(lr,threshold=1)
    data = sub_mod.fit(iris_data.data,iris_data.target).transform(iris_data.data)
    print("l1 和 l2 和并后处理的结果：",data[:3])

if __name__ == '__main__':
    main()
    print("---------------------------------------------------------")
    print("以上的结果说明了，使用不同的特征选择模型从而选取到不同的特征")