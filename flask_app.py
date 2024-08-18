from flask import Flask, render_template, request, send_file, flash, url_for, jsonify
from flask_api import status
import warnings

warnings.filterwarnings("ignore")
import random, string
import cv2
import json
from model import (
    ModelLandmarkRadio,
    ModelLandmarkProfile,
    ModelOrientation,
    ModelOrientationJulien,
    ModelClassifVisageFacePhoto,
    ModelClassifIntraCotePhoto,
    ModelClassifIntraFacePhoto,
    ModelClassifProfile,
)
from utils import load_diags, fill_diags

model_ori = ModelOrientation()
model_visage_face = ModelClassifVisageFacePhoto()
model_intra_cote = ModelClassifIntraCotePhoto()
model_intra_face = ModelClassifIntraFacePhoto()
model_ori_julien = ModelOrientationJulien()
model_landmark_profile = ModelLandmarkProfile()
model_landmark_radio = ModelLandmarkRadio()
model_profile = ModelClassifProfile()

# extensions
allowed_extensions = ["jpg", "jpeg", "png"]

# Flask
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classification_orientation", methods=["POST", "GET"])
def classif_orientation():

    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    orientation = model_ori(filepath)
    app.logger.info("Orientation Predicted !")

    all_classes = model_ori.all_classes()
    res = {k: 0 for k in all_classes}
    res[orientation] += 1

    return jsonify(res)


@app.route("/classification_orientation_julien", methods=["POST", "GET"])
def classif_orientation_julien():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    orientation = model_ori_julien(filepath)
    app.logger.info("Orientation Predicted !")

    all_classes = model_ori_julien.all_classes()
    res = {k: 0 for k in all_classes}
    res[orientation] += 1

    return jsonify(res)


@app.route("/classification_face_photo", methods=["POST", "GET"])
def classif_face_photo():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    all_preds = model_visage_face(filepath)
    app.logger.info("Classification Face Photo Predicted !")

    all_classes = model_visage_face.all_classes()
    res = {k: 0 for k in all_classes}
    for patho in all_preds:
        res[patho] += 1

    return jsonify(res)


@app.route("/classification_intra_cote", methods=["POST", "GET"])
def classif_intra_cote():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    all_preds = model_intra_cote(filepath)
    app.logger.info("Classification Intra Cote Predicted !")

    all_classes = model_intra_cote.all_classes()
    res = {k: 0 for k in all_classes}
    for patho in all_preds:
        res[patho] += 1

    return jsonify(res)


@app.route("/classification_intra_face", methods=["POST", "GET"])
def classif_intra_face():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    all_preds = model_intra_face(filepath)
    app.logger.info("Classification Intra Face Predicted !")

    all_classes = model_intra_face.all_classes()
    res = {k: 0 for k in all_classes}
    for patho in all_preds:
        res[patho] += 1

    return jsonify(res)


@app.route("/classification_profile", methods=["POST", "GET"])
def classification_profile():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    all_preds = model_profile(filepath)
    res = {k: 0 for k in all_preds}
    for patho in all_preds:
        res[patho] += 1
    app.logger.info("Classification Profile Predicted !")

    return jsonify(all_preds)


@app.route("/landmarks_profile", methods=["POST", "GET"])
def landmarks_profile():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    all_preds = model_landmark_profile(filepath)
    app.logger.info("Landmarks Profile Predicted !")

    print(all_preds)

    image = cv2.imread(filepath)

    for x, y in all_preds.values():
        image = cv2.circle(image, (x, y), 10, (255, 255, 255), -1)
    cv2.imwrite("test.png", image)

    return jsonify(all_preds)


@app.route("/landmarks_radio", methods=["POST", "GET"])
def landmarks_radio():

    filepath = ""
    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    all_preds = model_landmark_radio(filepath)
    app.logger.info("Landmarks Radio Predicted !")

    angles = model_landmark_radio.compute_all_angles(all_preds)
    app.logger.info("Angles Predicted !")

    return jsonify(angles)


@app.route("/diags", methods=["POST", "GET"])
def diags():

    if request.method != "POST":
        return render_template("pred.html", title="Orthosafe AI Prediction")

    if "img" not in request.files:
        app.logger.error("Cannot read input file")
        return "Cannot read input file", status.HTTP_400_BAD_REQUEST

    file = request.files["img"]
    extension = file.filename.split(".")[-1]
    # Check the file extension
    if extension.lower() not in allowed_extensions:
        app.logger.error("Bad input format")
        return "Bad input format", status.HTTP_400_BAD_REQUEST

    filepath = "./static/" + randomword(10) + "." + extension.lower()
    file.save(filepath)

    orientation = model_ori(filepath)
    diags = load_diags()
    app.logger.info("Json of diags loaded !")

    # Classif Visage face
    if (
        orientation == "full_face_with_smile"
        or orientation == "full_face_without_smile"
    ):

        app.logger.info(f"Orientation : {orientation}!")

        mapping = {
            0: ("face_shape", 2),
            1: ("face_shape", 3),
            2: ("face_shape", 1),
            3: ("face_type", 1),
            4: ("face_type", 3),
            5: ("face_type", 2),
            6: ("face_symmetry", 1),
            7: ("face_symmetry", 0),
            8: ("face_vertical_levels", 1),
            9: ("face_vertical_levels", 2),
            10: ("face_vertical_levels", 3),
            11: ("resting_labial", 1),
            12: ("resting_labial", 2),
            13: ("resting_labial", 3),
            14: ("face_chin_position", 1),
            15: ("face_chin_position", 2),
            16: ("face_chin_position", 3),
        }

        all_preds_face = model_visage_face(filepath)
        all_classes = model_visage_face.all_classes()
        res = {k: 0 for k in all_classes}
        for patho in all_preds_face:
            res[patho] += 1

        diags = fill_diags(res, diags, mapping)

        return jsonify(diags)

    # Classif Intra Face
    elif orientation == "front_view_of_teeth":

        app.logger.info(f"Orientation : {orientation}!")

        mapping = {
            0: ("inter_midpoints", 2),
            1: ("vertical_relationships", 2),
            2: ("transverse", 3),
            3: ("transverse", 2),
            4: ("vertical_relationships", 3),
            5: ("inter_midpoints", 1),
            6: ("maxillary", 2),
            7: ("transverse", 1),
            8: ("vertical_relationships", 1),
            9: ("maxillary", 1),
        }

        all_preds_intra_face = model_intra_face(filepath)
        all_classes = model_intra_face.all_classes()
        res = {k: 0 for k in all_classes}
        for patho in all_preds_intra_face:
            res[patho] += 1

        diags = fill_diags(res, diags, mapping)

        return jsonify(diags)

    # Classif Profile
    elif orientation == "face_in_profile":

        app.logger.info(f"Orientation : {orientation}!")

        mapping = {
            0: ("nasolabial_angle", 2),
            1: ("nasolabial_angle", 1),
            2: ("nasolabial_angle", 3),
            3: ("maxillary_position", 1),
            4: ("maxillary_position", 2),
            5: ("maxillary_position", 3),
            6: ("upper_lip_position", 1),
            7: ("upper_lip_position", 2),
            8: ("upper_lip_position", 3),
            9: ("lower_lip_position", 1),
            10: ("lower_lip_position", 2),
            11: ("lower_lip_position", 3),
            12: ("mandibular_floor_position", 1),
            13: ("mandibular_floor_position", 2),
            14: ("mandibular_floor_position", 3),
            15: ("profile_chin_position", 1),
            16: ("profile_chin_position", 2),
            17: ("profile_chin_position", 3),
        }

        all_preds_profile = model_profile(filepath)
        all_classes = model_intra_face.all_classes()
        res = {k: 0 for k in all_classes}
        for patho in all_preds_profile:
            res[patho] += 1

        diags = fill_diags(res, diags, mapping)

        return jsonify(diags)

    elif orientation == "left_view_of_teeth":

        app.logger.info(f"Orientation : {orientation}!")

        mapping = {
            0: ("Canine", "I"),
            1: ("Canine", "II", "partielle"),
            2: ("Canine", "II", "complète"),
            3: ("Canine", "III"),
            4: ("Molaire", "I"),
            5: ("Molaire", "II", "partielle"),
            6: ("Molaire", "II", "complète"),
            7: ("Molaire", "III"),
            8: "mandibular_curve_spee",
            9: "mandibular_curve_spee",
        }

        all_preds_intra_cote = model_intra_cote(filepath)
        all_classes = model_intra_cote.all_classes()
        res = {k: 0 for k in all_classes}
        for patho in all_preds_intra_cote:
            res[patho] += 1

        for i, (k, v) in enumerate(res.items()):

            if v == 1 and (i >= 0 and i <= 3):  # C'est une Canine
                app.logger.info("It's a canine !")

                diags["dentaires_canine"]["level_1"] = 1
                diags["dentaires_canine"]["gauche"]["level_2"] = 2
                diags["dentaires_canine"]["gauche"]["level_3"] = mapping[i][1]
                # diags['dentaires_canine']['gauche']['level_4'] = mapping[i][1] # PARTIELLE OR COMPLETE

            elif v == 1 and (i >= 4 and i <= 7):  # C'est une Molaire
                app.logger.info("It's a molaire !")

                diags["dentaires_molaire"]["level_1"] = 1
                diags["dentaires_molaire"]["gauche"]["level_2"] = 2
                diags["dentaires_molaire"]["gauche"]["level_3"] = mapping[i][
                    1
                ]  # I II OR III
                # diags['dentaires_molaire']['gauche']['level_4'] = mapping[i][1] # PARTIELLE OR COMPLETE

            elif v == 1 and (
                i > 7 and i <= 9
            ):  # Les courbes de spee sont dans le shéma api.
                app.logger.info("Spee informations !")

                classe = mapping[i]
                diags[classe] = 1

            # elif v == 1 and i > 9:   # ATTENTE RETOUR CALL API POUR RELATIONS TRANSVERSALES
            #     classe = mapping[i]
            #     diags[classe] = 1

        return jsonify(diags)

    elif orientation == "right_view_of_teeth":

        print("ORIENTATION : RIGHT VIEW OF TEETH")

        mapping = {
            0: ("Canine", "I"),
            1: ("Canine", "II", "partielle"),
            2: ("Canine", "II", "complète"),
            3: ("Canine", "III"),
            4: ("Molaire", "I"),
            5: ("Molaire", "II", "partielle"),
            6: ("Molaire", "II", "complète"),
            7: ("Molaire", "III"),
            8: "mandibular_curve_spee",
            9: "mandibular_curve_spee",
        }

        all_preds_intra_cote = model_intra_cote(filepath)
        all_classes = model_intra_cote.all_classes()
        res = {k: 0 for k in all_classes}
        for patho in all_preds_intra_cote:
            res[patho] += 1

        print("res", res)

        for i, (k, v) in enumerate(res.items()):

            if v == 1 and (i >= 0 and i <= 3):  # C'est une Canine
                diags["dentaires_canine"]["level_1"] = 1
                diags["dentaires_canine"]["droite"]["level_2"] = 1
                diags["dentaires_canine"]["droite"]["level_3"] = mapping[i][1]
                # diags['dentaires_canine']['droite']['level_4'] = mapping[i][1] # PARTIELLE OR COMPLETE

            elif v == 1 and (i >= 4 and i <= 7):  # C'est une Molaire
                diags["dentaires_molaire"]["level_1"] = 1
                diags["dentaires_molaire"]["droite"]["level_2"] = 1
                diags["dentaires_molaire"]["droite"]["level_3"] = mapping[i][
                    1
                ]  # I II OR III
                # diags['dentaires_molaire']['droite']['level_4'] = mapping[i][1] # PARTIELLE OR COMPLETE

            elif v == 1 and (
                i > 7 and i <= 9
            ):  # Les courbes de spee sont dans le shéma api.
                classe = mapping[i]
                diags[classe] = 1

        return jsonify(diags)

    elif orientation == "face_in_profile_radiographie":

        all_preds = model_landmark_radio(filepath)
        app.logger.info("Landmarks Radio Predicted !")

        angles = model_landmark_radio.compute_all_angles(all_preds)
        app.logger.info("Angles Predicted !")

        diags["anb"] = round(angles["ANB"], 2)
        diags["ans"] = round(angles["SNA"], 2)
        diags["snb"] = round(angles["SNB"], 2)

        return jsonify(diags)


if __name__ == "__main__":
    app.run(debug=True)
