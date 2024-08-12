from spectrascope.SpectraScope import *
from spectrascope.features import *
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import SPA
import seaborn as sns
from spectrascope.ss_classification import *


fnX = "test_files/X_testfile.npy"
fny = "test_files/y_testfile.npy"
X = np.load(fnX)
y = np.load(fny)

X = X[:, :10, :]
y = y[:10]

structures = np.load("test_files/structures.npy", allow_pickle=True)
mpids = np.load("test_files/mpids.npy")
mpids = list(mpids)

info = BinaryClassifications(
    structures, mpids, X=X, y=y, specnames=["pdf", "xanes"], yname="nn", random_seed=10
)


def test_update_defaults():
    info.update_structures_library_defaults()
    info.structure_library_to_df()
