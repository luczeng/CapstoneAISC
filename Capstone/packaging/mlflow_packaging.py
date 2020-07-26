import mlflow.pyfunc
from Capstone.modeling.resnet import initalize_resnet
import yaml
import torch


class resnet_wrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.net = initalize_resnet(2)
        self.net.type(torch.cuda.FloatTensor)

    def predict(self, context, model_input):
        example = torch.tensor(model_input.values)[None, None, :, :].type(torch.cuda.FloatTensor)
        probs = self.net(example)

        return probs.detach().cpu().numpy()


def package_model(mlflow_pyfunc_model_path, state_dict_path: str, env_path: str):

    # Artifacts (everything we want to package with the model)
    artifacts = {"state_dict": state_dict_path}

    # Load conda env
    with open(env_path, "r") as f:
        conda_env = yaml.load(env_path, Loader=yaml.FullLoader)

    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path,
        python_model=resnet_wrapper(),
        artifacts=artifacts,
        conda_env=conda_env,
        code_path=["Capstone"],
    )
