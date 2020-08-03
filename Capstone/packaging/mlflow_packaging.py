from Capstone.modeling.resnet import initalize_resnet
from Capstone.utils.nn_utils import load_checkpoint
import yaml
import torch
import torch.nn as nn
import mlflow.pyfunc


class resnet_wrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
            Load weights

            TODO: go in inference mode!
        """

        # Prepare architecture
        self.net = initalize_resnet(2)
        self.net.type(torch.cuda.FloatTensor)

        # Load weights
        print()
        # print("wut wut")
        load_checkpoint(context.artifacts["state_dict"], self.net)

        self.softmax = nn.Softmax(dim=1)

    def predict(self, context, model_input):
        """
            Runs inference on input folder and then applies softmax
        """

        # P Prepare data in torch format
        example = torch.tensor(model_input.values)[None, None, :, :].type(torch.cuda.FloatTensor)

        # Run inference
        probs = self.net(example)

        # Normalise
        probs = self.softmax(probs)

        return probs.detach().cpu().numpy()


def package_model(mlflow_pyfunc_model_path, state_dict_path: str, env_path: str):
    """

    """

    # Artifacts (everything we want to package with the model)
    artifacts = {"state_dict": state_dict_path}

    # Load conda env
    conda_env = yaml.load(env_path, Loader=yaml.FullLoader)

    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path,
        python_model=resnet_wrapper(),
        artifacts=artifacts,
        conda_env=conda_env,
        code_path=["Capstone"],
    )
