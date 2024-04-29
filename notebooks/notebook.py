import os

# permet de changer de working directory pour les notebooks afin de pouvoir importer les différentes classes nécessaires.
# plus d'information : https://stackoverflow.com/questions/35664972/how-to-change-working-directory-in-jupyter-notebook#:~:text=Type%20command%20cd%20%3Cyour%20new,with%20your%20new%20work%20directory.&text=Highly%20active%20question.


def define_path():
    PATH = os.getcwd()
    PATH = PATH.split(os.sep)
    PATH.remove(PATH[-1])
    PATH = os.sep.join(PATH)
    os.chdir(PATH)


if __name__ == "__main__":
    define_path()
    print(os.getcwd())
