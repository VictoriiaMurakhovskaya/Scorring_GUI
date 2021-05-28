from PyQt5 import QtWidgets as qw
from gui.main import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QCheckBox, QMessageBox, QStatusBar
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from numpy import hstack
from PyQt5.QtCore import QThread

models = ['Логістична регресія', 'XGBoost', 'Класифікатор LGBM',
          'Лінійний дискримінантний аналіз', 'Байєсівський класифікатор', 'Дерево рішень']

sk_lst = [LogisticRegression(random_state=0, C=0.01, n_jobs=6),
          xgb.XGBClassifier(n_jobs=6, verbosity=0),
          LGBMClassifier(n_jobs=6),
          LinearDiscriminantAnalysis(solver='lsqr'),
          GaussianNB(),
          DecisionTreeClassifier(criterion='entropy')]


class TheWindow(qw.QMainWindow):
    """
    Клас графічного інтерфейсу програми
    """

    def __init__(self):
        """
        Конструктор головного вікна програми
        """
        # ініціалізація основного вікна
        super(TheWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.on_exit)
        self.ui.pushButton_2.clicked.connect(self.on_evaluate)

        for model in models:
            widgetItem = QtWidgets.QListWidgetItem(self.ui.listWidget)
            self.ui.listWidget.setItemWidget(widgetItem, QCheckBox())
            widgetItem.setText('     ' + model)

        self.ui.listWidget.setStyleSheet("QListWidget {padding: 5px;}")

        self.ui.progressBar.setValue(0)
        self.ui.progressBar_2.setValue(0)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def on_exit(self):
        sys.exit(0)

    def checkedItems(self):
        for index in range(self.ui.listWidget.count()):
            item = self.ui.listWidget.item(index)
            box = self.ui.listWidget.itemWidget(item)
            yield 1 if box.checkState() > 0 else 0

    def on_evaluate(self):
        checkers = [checked for checked in self.checkedItems()]

        if sum(checkers) == 0:
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setText('Не обрано жодної моделі')
            error_dialog.setWindowTitle("Помилка")
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec()
        else:
            self.ui.pushButton_2.setDisabled(True)
            self.calc = Calculator(checkers, parent=self)
            self.calc.start()
            self.calc.start_calc.connect(self.show_message)
            self.calc.enable_button.connect(self.enable_button)
            self.calc.read_data.connect(self.clear_message)
            self.calc.finish_calc.connect(self.show_result)

    @QtCore.pyqtSlot(str)
    def show_message(self, message):
        self.statusBar.showMessage(message)

    @QtCore.pyqtSlot()
    def enable_button(self):
        self.ui.pushButton_2.setDisabled(False)

    @QtCore.pyqtSlot()
    def clear_message(self):
        self.statusBar.clearMessage()

    @QtCore.pyqtSlot(float, float)
    def show_result(self, accuracy, AUC):
        self.ui.progressBar.setValue(int(accuracy))
        self.ui.progressBar_2.setValue(int(AUC))


class Calculator(QThread):
    start_calc, read_data, finish_calc, enable_button = QtCore.pyqtSignal(str), QtCore.pyqtSignal(),\
                                                        QtCore.pyqtSignal(float, float), QtCore.pyqtSignal()

    def __init__(self, checkers, parent=None):
        QtCore.QThread.__init__(self, parent=parent)
        self.checkers = checkers
        self.df = None

    def fit_ensemble(self, models, X_train, X_val, y_train, y_val):
        meta_X = list()
        for model in models:
            model.fit(X_train, y_train)
            yhat = model.predict(X_val)
            yhat = yhat.reshape(len(yhat), 1)
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        blender = LogisticRegression()
        blender.fit(meta_X, y_val)
        return blender

    def predict_ensemble(self, models, blender, X_test):
        meta_X = list()
        for model in models:
            yhat = model.predict(X_test)
            yhat = yhat.reshape(len(yhat), 1)
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        return blender.predict(meta_X)

    def run(self):
        if sum(self.checkers) == 1:
            message = 'Використання базової моделі'
        else:
            message = 'Використання ансамблю моделей'
        self.start_calc.emit(message)

        self.df = pd.read_pickle('pyqt_data.pickle')

        y = self.df[['TARGET']].values
        X = self.df.drop(columns=['TARGET'])
        se = StandardScaler()
        X = normalize(se.fit_transform(X))
        self.read_data.emit()

        if sum(self.checkers) == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            modelN = self.checkers.index(True)
            clf = sk_lst[modelN]
            y_train = y_train.ravel()
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)
            AUC_score = np.floor(roc_auc_score(y_test, y_proba) * 100)
            Accuracy_score = np.floor(accuracy_score(y_test, y_pred) * 100)
            self.finish_calc.emit(Accuracy_score, AUC_score)
        else:
            selected_models = [sk_lst[i] for i in range(len(self.checkers)) if self.checkers[i] == 1]

            X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33,
                                                              random_state=1)

            blender = self.fit_ensemble(selected_models, X_train, X_val, y_train, y_val)
            yhat = self.predict_ensemble(selected_models, blender, X_test)
            score = np.round(accuracy_score(y_test, yhat) * 100)
            AUC_score = np.round(roc_auc_score(y_test, yhat) * 100)
            self.finish_calc.emit(score, AUC_score)
        self.enable_button.emit()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    app = qw.QApplication([])
    app.setStyle('Fusion')
    application = TheWindow()
    application.show()
    sys.exit(app.exec_())