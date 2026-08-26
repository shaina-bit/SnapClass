import streamlit as st
from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
import numpy as np
from src.pipelines.face_pipeline import (
    predict_attendance,
    get_face_embeddings,
    train_classifier
)
from src.pipelines.voice_pipeline import get_voice_embeddings
from src.database.db import get_all_students, create_student,get_student_subject,get_student_attendance, unenroll_student_to_subject
import time
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card

def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:
        header_dashboard()

    with c2:

        st.subheader(
            f"Welcome, {student_data['name']}"
        )

        if st.button(
            "Logout",
            type="secondary",
            key="student_logout_btn",
            shortcut="control+backspace"
        ):

            st.session_state["is_logged_in"] = False

            if "student_data" in st.session_state:
                del st.session_state["student_data"]

            st.rerun()

    st.space()

    c1, c2 = st.columns(2)
    with c1:
        st.header('Your Enrolled Subjects')
    with c2:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()

    st.divider()

    with st.spinner('Loading your enrolled subjects...'):
        subjects = get_student_subject(student_id)
        logs = get_student_attendance(student_id)

    stats_map ={}

    for log in logs:
        sid= log['subject_id']

        if sid not in stats_map:
            stats_map[sid] = {"total": 0,"attended":0}
        stats_map[sid]['total'] +=1

        if logs.get('is_present'):
            stats_map[sid]['attended']+=1

    cols = st.columns(2)
    for i, sub_node in enumerate(subjects):
        sub = sub_node['subjects']
        sid = sub['subject_id']

        stats = stats_map.get(sid, {"total": 0,"attended":0})
        def unenroll_button():
            if st.button('Unenroll from this course', type='tertiary', width='stretch',icon=':material/delete_forever:'):
                unenroll_student_to_subject(student_id, sid)
                st.toast(f"Unenroll from {sub['name']} successfully!")
                st.rerun()
        with cols[i%2]:
            subject_card(
                name = sub['name'],
                code = sub['subject_code'],
                section = sub['section'],
                stats = [
                    ('📅', 'Total', stats['total']),
                    ('✅', 'Attended', stats['attended']),
                ],
                footer_callback= unenroll_button
            )
    footer_dashboard()


def student_screen():
    style_background_dashboard()
    style_base_layout()

    # --------------------------------------------------
    # ALREADY LOGGED IN
    # --------------------------------------------------

    if "student_data" in st.session_state:
        student_dashboard()
        return

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:
        header_dashboard()

    with c2:
        if st.button(
            "Go back to Home",
            type="secondary",
            key="loginbackbtn",
            shortcut="control+backspace"
        ):
            st.session_state["login_type"] = None
            st.rerun()

    # --------------------------------------------------
    # PAGE TITLE
    # --------------------------------------------------

    st.header(
        "Login using FaceID",
        text_alignment="center"
    )

    st.space()
    st.space()

    # This controls whether the registration form appears
    show_registration = False

    # --------------------------------------------------
    # CAMERA
    # --------------------------------------------------

    photo_source = st.camera_input(
        "Position your face in the center"
    )

    # --------------------------------------------------
    # PROCESS PHOTO
    # --------------------------------------------------

    if photo_source:

        st.success("📸 Photo captured!")

        img = np.array(
            Image.open(photo_source)
        )


        # Run face recognition
        with st.spinner("🤖 AI is scanning..."):

            st.write(
                "Running face recognition..."
            )

            detected, all_ids, num_faces = predict_attendance(
                img
            )
        # --------------------------------------------------
        # NO FACE
        # --------------------------------------------------

        if num_faces == 0:

            st.warning(
                "Face not found! Please position your face "
                "inside the camera frame and try again."
            )

        # --------------------------------------------------
        # MULTIPLE FACES
        # --------------------------------------------------

        elif num_faces > 1:

            st.warning(
                "Multiple faces found! Please make sure "
                "only one person is visible."
            )

        # --------------------------------------------------
        # EXACTLY ONE FACE
        # --------------------------------------------------

        else:

            # ----------------------------------------------
            # FACE WAS RECOGNIZED
            # ----------------------------------------------

            if detected:

                student_id = list(
                    detected.keys()
                )[0]

                all_students = get_all_students()

                student = next(
                    (
                        s
                        for s in all_students
                        if s["student_id"] == student_id
                    ),
                    None
                )

                # ------------------------------------------
                # STUDENT FOUND
                # ------------------------------------------

                if student:

                    st.session_state.is_logged_in = True
                    st.session_state.user_role = "student"
                    st.session_state.student_data = student

                    st.toast(
                        f"Welcome Back {student['name']}!"
                    )

                    time.sleep(1)

                    st.rerun()

                # ------------------------------------------
                # FACE MATCHED BUT DATABASE RECORD MISSING
                # ------------------------------------------

                else:

                    st.warning(
                        "Face was recognized, but the "
                        "student record could not be found."
                    )

            # ----------------------------------------------
            # FACE NOT RECOGNIZED
            # ----------------------------------------------

            else:

                st.info(
                    "Face not recognized! "
                    "You might be a new student."
                )

                # This makes the registration form appear
                show_registration = True

    # ==================================================
    # REGISTRATION FORM
    # ==================================================

    if show_registration:

        with st.container(border=True):

            st.header("Register New Profile")

            # ----------------------------------------------
            # NAME
            # ----------------------------------------------

            new_name = st.text_input(
                "Enter your name",
                placeholder="E.g. Shaina Noushad"
            )

            # ----------------------------------------------
            # VOICE ENROLLMENT
            # ----------------------------------------------

            st.subheader(
                "Optional: Voice Enrollment"
            )

            st.info(
                "Enroll your voice for voice-only attendance."
            )

            audio_data = None

            try:

                audio_data = st.audio_input(
                    "Record a phrase like: "
                    "I am present, my name is Shaina."
                )

            except Exception as e:

                st.error(
                    f"Audio data failed: {e}"
                )

            # ----------------------------------------------
            # CREATE ACCOUNT
            # ----------------------------------------------

            if st.button(
                "Create Account",
                type="primary"
            ):

                # ------------------------------------------
                # NAME VALIDATION
                # ------------------------------------------

                if not new_name.strip():

                    st.warning(
                        "Please enter your name!"
                    )

                else:

                    with st.spinner(
                        "Creating profile..."
                    ):

                        # ----------------------------------
                        # GET FACE EMBEDDING
                        # ----------------------------------

                        img = np.array(
                            Image.open(photo_source)
                        )

                        encodings = get_face_embeddings(
                            img
                        )

                        if encodings:

                            face_emb = encodings[0].tolist()

                            # ------------------------------
                            # GET VOICE EMBEDDING
                            # ------------------------------

                            voice_emb = None

                            if audio_data:

                                voice_emb = get_voice_embeddings(
                                    audio_data.read()
                                )

                            # ------------------------------
                            # CREATE STUDENT
                            # ------------------------------

                            response_data = create_student(
                                new_name.strip(),
                                face_embedding=face_emb,
                                voice_embedding=voice_emb
                            )

                            # ------------------------------
                            # SUCCESS
                            # ------------------------------

                            if response_data:

                                # Refresh face recognition model
                                train_classifier()

                                # Log student in
                                st.session_state.is_logged_in = True
                                st.session_state.user_role = "student"
                                st.session_state.student_data = (
                                    response_data[0]
                                )

                                st.toast(
                                    f"Profile Created! "
                                    f"Hi {new_name.strip()}!"
                                )

                                time.sleep(1)

                                st.rerun()

                            # ------------------------------
                            # DATABASE FAILURE
                            # ------------------------------

                            else:

                                st.error(
                                    "Failed to create student "
                                    "profile."
                                )

                        # ------------------------------
                        # FACE EMBEDDING FAILURE
                        # ------------------------------

                        else:

                            st.error(
                                "Couldn't capture your facial "
                                "features for registration."
                            )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    footer_dashboard()