import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).parent.parent))
from src.data import make_synthetic;from src.model import fit_and_evaluate;from src.core import concordance_index
def test_data():d=make_synthetic(300);assert d["n_samples"]==300
def test_ci():assert concordance_index([5,10,15],[1,1,1],[3,2,1])>0.5
def test_fit():d=make_synthetic(400);m,met=fit_and_evaluate(d);assert met["c_index"]>0.3
if __name__=="__main__":test_data();test_ci();test_fit();print("OK")
