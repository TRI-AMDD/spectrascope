from spectrascope.SpectraScope import *
from spectrascope.Features import *
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import SPA
import seaborn as sns
from spectrascope.SSClassification import *

test_dir = os.path.join(os.path.dirname(__file__), "test_files")
fnX = os.path.join(test_dir, "X_testfile.npy")
fny = os.path.join(test_dir, "y_testfile.npy")
X = np.load(fnX)
y = np.load(fny)

X = X[:, :10, :]
y = y[:10]

structures = np.load(os.path.join(test_dir, "structures.npy"), allow_pickle=True)
mpids = np.load(os.path.join(test_dir, "mpids.npy"))
mpids = list(mpids)

info = BinaryClassifications(
    structures, mpids, X=X, y=y, specnames=["pdf", "xanes"], yname="nn", random_seed=10
)


def test_update_defaults():
    info.update_structures_library_defaults()
    info.structure_library_to_df()
