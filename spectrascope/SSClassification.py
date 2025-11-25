from spectrascope.SpectraScope import *
from spectrascope.Features import *
from spectrascope.StructurePropertyFeatures import *
from collections import OrderedDict
from tqdm import tqdm_notebook, tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spectrascope.utils import *

from scipy.spatial.distance import cdist
from matminer.featurizers.structure import SiteStatsFingerprint

__docformat__ = "numpy"

class BinaryClassifications(SpectraScope):

    def __init__(
        self, structures, mpids, nn_algo=CrystalNN(), generate_dict=True, **kwargs
    ):

        super().__init__(**kwargs)

        self.structures = structures
        self.nn_algo = nn_algo
        self.structures_library = OrderedDict()
        self.mpids = mpids
        self.entropy_dict = {}

        if generate_dict:
            self.generate_dict()

    def generate_dict(self):
        nn_library = {}

        nn_algo = self.nn_algo
        self.cns_present = {}
        self.els_present = {}
        self.oxi_states_present = {}
        cnnf = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")

        self.crystal_systems_present = {
            "triclinic": 0,
            "monoclinic": 0,
            "orthorhombic": 0,
            "tetragonal": 0,
            "trigonal": 0,
            "hexagonal": 0,
            "cubic": 0,
        }
        crystal_systems = list(self.crystal_systems_present.keys())
        self.nns_present = []
        self.space_groups_present = {}

        for i, struct in tqdm_notebook(enumerate(self.structures)):
            self.structures_library[self.mpids[i]] = {"structure": struct.as_dict()}

            temp = identify_crystal_system(struct)
            is_true = np.argwhere(list(temp.values()))[0][0]
            true_key = crystal_systems[is_true]
            self.crystal_systems_present[true_key] += 1

            space_group = SpacegroupAnalyzer(struct).get_space_group_symbol()
            self.present_in_dict(self.space_groups_present, space_group)

            # self.structures_library[self.mpids[i]][f'is_{space_group}'] = True
            for crystal_systems_key in temp:
                self.structures_library[self.mpids[i]][crystal_systems_key] = temp[
                    crystal_systems_key
                ]

            struct.add_oxidation_state_by_guess()
            structure_vector = cnnf.featurize(struct)
            self.structures_library[self.mpids[i]][
                "structure_vector"
            ] = structure_vector

            indices = range(len(struct))
            site_dict = {}
            for n in indices:
                species_string = struct[n].specie.element.__str__()
                nn_info = nn_algo.get_nn_info(struct, n)

                nn_list = []
                mean_dist = []
                for j, neighbor in enumerate(nn_info):
                    neighbor_string = neighbor["site"].specie.element.__str__()
                    distance = struct.get_distance(n, neighbor["site_index"])
                    mean_dist.append(distance)
                    nn_list.append((neighbor_string, distance))

                site_dict[n] = {species_string: nn_list}
                nn_mean = np.nanmean(mean_dist)
                oxi_state = struct[n].specie.oxi_state
                cn = len(nn_list)

                site_dict[n]["nn_mean"] = nn_mean
                site_dict[n]["oxi_state"] = oxi_state
                site_dict[n]["cn"] = cn

                self.nns_present.append(nn_mean)

                self.present_in_dict(self.oxi_states_present, oxi_state)
                self.present_in_dict(self.els_present, species_string)
                self.present_in_dict(self.cns_present, cn)

            nn_library[struct.__str__()] = site_dict

        self.nn_library = nn_library

    def present_in_dict(self, my_dict, key):
        if key not in my_dict:
            my_dict[key] = 1
        else:
            my_dict[key] += 1

    def plot_hist(
        self, my_list, xaxis_title, yaxis_title, title, nbins, return_fig=False
    ):
        fig = px.histogram(my_list, nbins)
        fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title, title=title)

        if return_fig:
            return fig
        else:
            fig.show()

    def plot_bar(self, labels, values, return_fig=False):
        fig = px.bar(x=labels, y=values)
        fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title, title=title)
        fig.show()

        if return_fig:
            return fig
        else:
            fig.show()

    def plot_nns(self):
        self.plot_hist(
            self.nns_present, "nn mean distance, per site", "counts", nbins=30
        )

    def plot_my_dict(self, my_dict, xaxis_title, yaxis_title, title, id2, id1):
        labels, values = sort_get_value(my_dict, id2, id1)
        self.plot_bar(
            labels,
            values,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            title=title,
        )

    def plot_all(self, return_fig=False, update_defaults=True):
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Elements",
                "Mean NN Distance",
                "CN",
                "Oxidation State",
                "Space Group",
                "Crystal System",
            ),
        )

        labels, values = sort_get_value(self.els_present, 20, 2)
        fig.append_trace(go.Bar(x=labels, y=values, name="el"), row=1, col=1)
        fig.add_vrect(
            x0=-0.5,
            x1=10,
            row=1,
            col=1,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
            annotation_text="default",
        )

        labels, values = sort_get_value(self.crystal_systems_present, 6, 0)
        fig.append_trace(
            go.Bar(x=labels, y=values, name="crystal system"), row=2, col=3
        )
        fig.add_vrect(
            x0=-0.5,
            x1=6,
            row=2,
            col=3,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
            annotation_text="default",
        )

        labels, values = sort_get_value(self.cns_present, 10, 0)
        fig.append_trace(go.Bar(x=labels, y=values, name="cn"), row=1, col=3)
        fig.add_vrect(
            x0=-0.5,
            x1=4,
            row=1,
            col=3,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
            annotation_text="default",
        )

        labels, values = sort_get_value(self.oxi_states_present, 10, 1)
        fig.append_trace(
            go.Bar(x=labels, y=values, name="oxidation state"), row=2, col=1
        )
        fig.add_vrect(
            x0=-0.5,
            x1=6,
            row=2,
            col=1,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
            annotation_text="default",
        )

        labels, values = sort_get_value(self.space_groups_present, 20, 0)
        fig.append_trace(go.Bar(x=labels, y=values, name="space group"), row=2, col=2)
        fig.add_vrect(
            x0=-0.5,
            x1=10,
            row=2,
            col=2,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
            annotation_text="default",
        )

        labels, values = sort_get_value(self.space_groups_present, 20, 0)
        fig.append_trace(
            go.Histogram(x=self.nns_present, nbinsx=30, name="nn"), row=1, col=2
        )
        fig.add_vrect(
            x0=1.8,
            x1=2.2,
            row=1,
            col=2,
            fillcolor="green",
            opacity=0.25,
            line_width=0,
            annotation_text="default",
        )

        if update_defaults:
            self.update_structures_library_defaults()

        if return_fig:
            return fig
        else:
            fig.show()

    def chosen_keys(self, my_dict, id2, id1):
        labels, values = sort_get_value(my_dict, id2, id1)
        return labels

    def update_structures_library(
        self, my_dict, function, chosen=None, id2=10, id1=0, **kwargs
    ):
        mpids = self.mpids
        args = function.__code__.co_varnames[: function.__code__.co_argcount]
        args = [arg for arg in args if arg != "structure" and arg != "return_label"]

        if not chosen:
            chosen = self.chosen_keys(my_dict, id2, id1)
        for i, struct in tqdm_notebook(enumerate(self.structures)):
            for key in chosen:
                if (
                    type(key) != tuple
                    and type(key) != list
                    and type(key) != type(np.array([]))
                ):
                    key = [key]
                for j, k in enumerate(key):
                    kwargs[args[j]] = k
                value, label = function.__call__(struct, **kwargs, return_label=True)
                if mpids[i] not in self.structures_library:
                    self.structures_library[mpids[i]] = {}
                self.structures_library[mpids[i]][label] = value
                # print(mpids[i])
            # print(pd.DataFrame(self.structures_library).T.head())

    def update_structures_library_defaults(self):
        self.update_structures_library(
            self.els_present, is_element_present, None, id2=10, id1=2
        )
        self.update_structures_library(
            self.cns_present, is_cn_present, None, id2=5, id1=0
        )
        self.update_structures_library(
            self.oxi_states_present, is_oxidation_state_present, None, id2=5, id1=1
        )
        self.update_structures_library(
            self.space_groups_present, is_in_space_group, None, id2=10, id1=0
        )
        self.update_structures_library(
            self.nns_present, is_nn_distance_present, chosen=[(1.2, 2)]
        )

    def structure_library_to_df(self):
        df = pd.DataFrame(self.structures_library).T
        self.structures_df = df

        # vog = np.array([f for f in self.structures_df['structure_vector']])
        # self.og_similarity = self.av_distance(vog)
        self.ground_truth_df = self.structures_df[
            [col for col in self.structures_df if "structure" not in col]
        ]
        self.ground_truth_similarity = np.zeros(
            [len(self.ground_truth_df), len(self.ground_truth_df)]
        )
        self.ground_truth = self.ground_truth_df.to_numpy()

        for i, s1 in enumerate(self.ground_truth):
            J = tamimoto(s1, self.ground_truth)
            self.ground_truth_similarity[i] = J.ravel()

    def av_distance(self, v):
        distances = np.triu(cdist(v, v)).ravel()
        nonzero_distances = distances.nonzero()
        av_dist = distances[nonzero_distances].mean()
        return av_dist

    def do_one_classification(
        self,
        Xattr,
        yattr,
        test_size=0.25,
        savef=None,
        use_pipeline=False,
        scale_X=True,
        **kwargs
    ):
        if not hasattr(self, "structures_df"):
            self.structure_library_to_df()

        self.add_y(yattr, self.structures_df[yattr].astype(int))
        self.split_y(yattr, test_size)
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import NotFittedError

        estimator = kwargs["estimator"]
        # try:
        #     check_is_fitted(estimator)
        #     print('already fitted')
        # except NotFittedError as exc:
        #     print('not fitted')

        self.do_sk(
            Xattr,
            yattr,
            test_size,
            use_pipeline=use_pipeline,
            scale_X=scale_X,
            **kwargs
        )
        # self.do_LCEN(Xattr, yattr, test_size, savef, scale_X = scale_X, **kwargs)

        model_name = estimator.__str__().split("(")[0]
        fit_dict = getattr(self, Xattr + "_" + yattr + "_" + model_name)
        P = fit_dict["P"]
        full_prediction = P.predict(getattr(self, Xattr))

        filtered_df = self.structures_df[full_prediction == 1]
        # v = np.array([f for f in filtered_df['structure_vector']])
        # av_dist = self.av_distance(v)

        this_entropy = arr_entropy(full_prediction)

        fit_dict["entropy"] = this_entropy
        fit_dict["filtered_df"] = filtered_df
        fit_dict["full_prediction"] = full_prediction
        if model_name not in self.entropy_dict:
            self.entropy_dict[model_name] = {Xattr + "_" + yattr: this_entropy}
        else:
            self.entropy_dict[model_name][Xattr + "_" + yattr] = this_entropy

        return full_prediction

    def do_all_classification(
        self,
        Xattr,
        yattrs: list,
        test_size=0.25,
        savef=None,
        use_pipeline=False,
        scale_X=True,
        **kwargs
    ):
        masks = np.zeros([len(getattr(self, Xattr)), len(yattrs)])
        from sklearn.base import clone

        for i, yattr in enumerate(yattrs):
            kwargs["estimator"] = clone(kwargs["estimator"])
            full_prediction = self.do_one_classification(
                Xattr, yattr, test_size, savef, use_pipeline, scale_X=scale_X, **kwargs
            )
            masks[:, i] = full_prediction

        mask_df = pd.DataFrame(masks, columns=yattrs)
        #calculating the entropy of the different yattr columns
        entropies = arr_entropy(mask_df)
        best_order = np.argsort(entropies)
        yattr_ordered = np.array(yattrs)[best_order]

        sorted_truth = self.ground_truth_df[yattr_ordered]
        similarity_results = np.zeros([len(yattrs), masks.shape[0], masks.shape[0]])
        for i, yattr in enumerate(yattr_ordered):
            selected = yattr_ordered[: i + 1]
            similarity_result = self.similarity_matrix(
                mask_df[selected].to_numpy(), sorted_truth.to_numpy()[:, : i + 1]
            )
            similarity_results[i] = similarity_result

        return similarity_results, yattr_ordered

    def similarity_matrix(self, mat1, mat2):
        assert mat1.shape[0] == mat2.shape[0]
        result_mat = np.zeros([mat1.shape[0], mat1.shape[0]])

        for i, s1 in enumerate(mat1):
            J = tamimoto(s1, mat2)
            result_mat[i] = J.ravel()
        return result_mat

    def plot_structure_entropy(self, **plotly_kwargs):
        temp = pd.DataFrame(self.entropy_dict)
        if len(self.entropy_dict) == 0:
            print("you havent done any fitting yet")

        self.plot_df_heatmap(temp, **plotly_kwargs)

    def find_similar_structures(
        self, spec_id, Xattr, selected_yattrs, model_name, top=10
    ):
        specs = getattr(self, Xattr)
        spec = specs[spec_id]

        spec_classifications = []
        for selected in selected_yattrs:
            fit_df = getattr(self, Xattr + "_" + selected + "_" + model_name)
            spec_classifications.append(fit_df["full_prediction"][spec_id])

        s1 = np.array(spec_classifications)
        mat2 = self.ground_truth_df[selected_yattrs].to_numpy()
        J = tamimoto(s1, mat2)
        top10 = np.argsort(J)[::-1][:top]
        return spec_classifications, np.array(self.mpids)[top10]

    def fit_and_find_similar_structures(self, nselect=-1):
        pass
