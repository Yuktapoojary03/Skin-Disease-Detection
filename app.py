from flask import Flask, request, render_template, redirect,url_for, session
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import sqlite3


app = Flask(__name__)

# Load the trained model
model = load_model(r'E:\Skin disease detection\models\skin_disease_model_best.keras')

# Define the class labels
classes = ['Acanthosis-nigricans', 'Amyloidosis', 'BA-cellulitis', 'Bechet', 'Birt hogg dube 1', 
           'Birt hogg dube 2', 'Erythema-infectiosum-1', 'Erythema-infectiosum-2', 'FU-athlete-foot',
           'FU-nail-fungus', 'FU-ringworm', 'Fixed-drug-eruption', 'Hand-foot-mouth-disease',
           'Hypothyroidism-1', 'Hypothyroidism-2', 'Kawasaki-syndrome', 'Lipoid-proteinosis',
           'Neurofibromatosis', 'Nevoxantho endothelioma', 'Scarlet-fever', 'Xanthomas', 'Scabies,Lyme Disease and other infestations and Bites', 
           'Warts, Molluscum and other Viral Infections','Urticaria Disease','Seborrheic Keratoses and other Benign Tumors',
           'Psoriasis,Lichen Planus and Related Diseases','Poison Ivy and other Contact Dermatitis','Melanoma,Skin Cancer,Nevi and Moles',
           'Lupus and other Connective Tissue Diseases','Light Diseases and Disorders of Pigmentation','Hair Loss,Alopecia and other Hair Diseases',
           'Systemic Disease','Exanthems and Drug Eruptioms','Eczema','Cellulitis,Impetigo and other Bacterial infections','Bullous Disease','Atopic Dermatitis',
           'Actinic Keratosis,Basal Cell Carcinoma and other Malignant Lesions','Acne and Rosacea','Vascular Tumors','Vasculitis Disease']

# Define Do's and Don'ts for each disease
disease_guidelines = {
    'Acanthosis-nigricans': {
        'do': ['Maintain a healthy weight', 'Follow a balanced diet', 'Consult a dermatologist for treatment'],
        'dont': ['Ignore persistent skin changes', 'Use harsh skin products', 'Skip medical checkups']
    },
    'Amyloidosis': {
        'do': ['Stay hydrated', 'Follow prescribed medications', 'Seek regular medical advice'],
        'dont': ['Ignore swelling or fatigue', 'Delay diagnosis', 'Consume excessive salt']
    },
    'BA-cellulitis': {
        'do': ['Keep wounds clean and covered', 'Take prescribed antibiotics promptly', 'Monitor for spreading redness'],
        'dont': ['Ignore signs of infection like fever', 'Scratch affected areas', 'Delay seeking treatment']
    },
    'Bechet': {
        'do': ['Maintain oral hygiene', 'Follow your doctors medication plan', 'Protect eyes from UV rays'],
        'dont': ['Ignore ulcers or inflammation', 'Miss regular health checkups', 'Use over-the-counter medications without consulting a doctor']
    },
    'Birt hogg dube 1': {
        'do': ['Regularly check for skin growths', 'Consult a dermatologist for unusual bumps', 'Maintain a healthy lifestyle'],
        'dont': ['Ignore lumps or bumps', 'Delay consulting specialists', 'Use non-prescribed treatments']
    },
    'Birt hogg dube 2': {
        'do': ['Monitor for unusual skin or lung symptoms', 'Follow genetic counseling advice', 'Undergo regular health screenings'],
        'dont': ['Ignore family history of the condition', 'Delay medical evaluations', 'Self-diagnose skin lesions']
    },
    'Erythema-infectiosum-1': {
        'do': ['Rest and stay hydrated', 'Follow prescribed medications', 'Wash hands frequently'],
        'dont': ['Ignore flu-like symptoms', 'Expose others if contagious', 'Delay treatment for joint pain']
    },
    'Erythema-infectiosum-2': {
        'do': ['Get plenty of rest', 'Practice good hygiene', 'Avoid close contact with vulnerable people'],
        'dont': ['Underestimate its severity', 'Ignore skin rashes', 'Delay seeking medical care for complications']
    },
    'FU-athlete-foot': {
        'do': ['Keep feet dry and clean', 'Wear breathable footwear', 'Use antifungal treatments as prescribed'],
        'dont': ['Share footwear or towels', 'Walk barefoot in public areas', 'Ignore persistent itching or redness']
    },
    'FU-nail-fungus': {
        'do': ['Trim and clean nails regularly', 'Use antifungal medications', 'Consult a dermatologist for severe cases'],
        'dont': ['Wear tight or non-breathable shoes', 'Ignore discoloration or thickened nails', 'Stop treatment prematurely']
    },
    'FU-ringworm': {
        'do': ['Use prescribed antifungal creams', 'Keep affected areas clean and dry', 'Wash clothes and bedding frequently'],
        'dont': ['Share personal items like towels', 'Scratch affected areas', 'Delay treatment']
    },
    'Fixed-drug-eruption': {
        'do': ['Inform your doctor about drug reactions', 'Avoid triggering medications', 'Use prescribed skin treatments'],
        'dont': ['Continue using suspected medications', 'Expose affected skin to irritants', 'Ignore recurring rashes']
    },
    'Hand-foot-mouth-disease': {
        'do': ['Rest and stay hydrated', 'Wash hands frequently', 'Use soothing treatments for mouth sores'],
        'dont': ['Share food or utensils', 'Ignore dehydration symptoms', 'Expose others to the infection']
    },
    'Hypothyroidism-1': {
        'do': ['Follow your prescribed medication schedule', 'Maintain a balanced diet with iodine', 'Get regular thyroid function tests'],
        'dont': ['Ignore symptoms like fatigue or weight gain', 'Stop medication without consulting a doctor', 'Consume excessive goitrogens (raw cruciferous vegetables)']
    },
    'Hypothyroidism-2': {
        'do': ['Follow your doctors advice', 'Stay physically active', 'Monitor for changes in symptoms'],
        'dont': ['Ignore signs of depression or cold sensitivity', 'Self-medicate', 'Skip regular checkups']
    },
    'Kawasaki-syndrome': {
        'do': ['Seek immediate medical attention', 'Follow treatment plans carefully', 'Monitor for cardiac complications'],
        'dont': ['Delay treatment for fever or rash', 'Ignore peeling skin or red eyes', 'Miss follow-up appointments']
    },
    'Lipoid-proteinosis': {
        'do': ['Maintain a healthy lifestyle', 'Follow treatment plans for symptom management', 'Seek regular dermatological and neurological evaluations'],
        'dont': ['Ignore progressive symptoms', 'Delay specialist consultations', 'Attempt unverified treatments']
    },
    'Neurofibromatosis': {
        'do': ['Get regular screenings for complications', 'Monitor skin changes and growths', 'Seek genetic counseling if necessary'],
        'dont': ['Ignore persistent pain or neurological symptoms', 'Miss follow-up appointments', 'Attempt to remove growths yourself']
    },
    'Nevoxantho endothelioma': {
        'do': ['Monitor for changes in skin lesions', 'Consult a dermatologist regularly', 'Follow treatment recommendations'],
        'dont': ['Ignore growing or painful lesions', 'Delay seeking medical advice', 'Use unproven remedies']
    },
    'Scarlet-fever': {
        'do': ['Take antibiotics as prescribed', 'Rest and stay hydrated', 'Maintain good hygiene to prevent spread'],
        'dont': ['Stop antibiotics prematurely', 'Expose others to the infection', 'Ignore signs of worsening symptoms']
    },
    'Xanthomas': {
        'do': ['Manage cholesterol levels with a healthy diet', 'Follow prescribed treatments', 'Consult a doctor for skin growth evaluation'],
        'dont': ['Ignore underlying health issues', 'Use harsh skin treatments on lesions', 'Delay lifestyle changes for cholesterol management']
    },
    'Scabies,Lyme Disease and other Infestations and Bites': { 
        'do': ['Maintain proper hygiene', 'Use prescribed medications', 'Wash bedding and clothes in hot water'],
        'dont': ['Scratch the affected area', 'Share personal items', 'Delay treatment']
    },
    'Warts, Molluscum and other Viral Infections': {
        'do': ['Keep the area clean and dry', 'Use doctor-recommended treatments', 'Cover warts to prevent spreading'],
        'dont': ['Pick or scratch warts', 'Share towels or personal items', 'Ignore growing or painful warts']
    },
    'Urticaria Disease': {
        'do': ['Identify and avoid triggers', 'Use antihistamines as prescribed', 'Apply cool compresses for relief'],
        'dont': ['Scratch or rub the skin', 'Ignore severe swelling or breathing difficulties', 'Expose skin to extreme heat or cold']
    },
    'Seborrheic Keratoses and other Benign Tumors': {
        'do': ['Consult a dermatologist for skin changes', 'Maintain sun protection', 'Monitor for changes in size or color'],
        'dont': ['Pick or scratch the lesion', 'Attempt home removal', 'Ignore new or changing growths']
    },
    'Psoriasis,Lichen Planus and Related Diseases': {
        'do': ['Use moisturizers regularly', 'Manage stress with relaxation techniques', 'Follow your dermatologists treatment plan'],
        'dont': ['Smoke or consume alcohol excessively', 'Ignore skin infections', 'Miss follow-up appointments']
    },
    'Poison Ivy and other Contact Dermatitis': {
        'do': ['Wash skin immediately with soap and water', 'Apply calamine lotion or hydrocortisone', 'Wear protective clothing when outdoors'],
        'dont': ['Scratch the rash', 'Burn poison ivy plants', 'Ignore severe reactions']
    },
    'Melanoma,Skin Cancer,Nevi and Moles': {
        'do': ['Perform regular skin checks', 'Use sunscreen with SPF 30 or higher', 'Seek immediate medical advice for suspicious moles'],
        'dont': ['Ignore changing moles', 'Use tanning beds', 'Delay dermatologist visits']
    },
    'Lupus and other Connective Tissue Diseases': {
        'do': ['Wear sun protection', 'Manage stress levels', 'Follow your doctors treatment plan'],
        'dont': ['Stay in the sun for extended periods', 'Ignore fatigue or joint pain', 'Stop medications without doctor approval']
    },
    'Light Diseases and Disorders of Pigmentation': {
        'do': ['Use sunscreen regularly', 'Stay hydrated', 'Consult a dermatologist for treatment options'],
        'dont': ['Ignore skin changes', 'Expose skin to the sun without protection', 'Use unproven treatments']
    },
    'Hair Loss,Alopecia and other Hair Diseases': {
        'do': ['Follow the dermatologists treatment plan', 'Use gentle hair care products', 'Maintain a nutritious diet'],
        'dont': ['Pull or twist hair', 'Use harsh chemicals or heat treatments', 'Ignore sudden or excessive hair loss']
    },
    'Systemic Disease': {
        'do': ['Manage underlying conditions', 'Take prescribed medications', 'Follow a healthy lifestyle'],
        'dont': ['Ignore unusual symptoms', 'Miss regular health checkups', 'Self-medicate without diagnosis']
    },
    'Exanthems and Drug Eruptions': {
        'do': ['Stay hydrated', 'Take fever-reducing medications as prescribed', 'Isolate if contagious'],
        'dont': ['Ignore symptoms of infection', 'Delay medical attention', 'Expose others to contagious rashes']
    },
    'Eczema': {
        'do': ['Moisturize daily', 'Use gentle, fragrance-free products', 'Apply prescribed medications'],
        'dont': ['Take long, hot showers', 'Scratch or rub the skin', 'Expose skin to known allergens']
    },
    'Cellulitis,Impetigo and other Bacterial Infections': {
        'do': ['Clean and cover any wounds', 'Take prescribed antibiotics', 'Elevate the affected area if possible'],
        'dont': ['Ignore redness, swelling, or warmth', 'Delay medical treatment', 'Stop antibiotics prematurely']
    },
    'Bullous Disease': {
        'do': ['Keep blisters clean and dry', 'Use prescribed medications', 'Protect the skin from injury'],
        'dont': ['Pop or burst blisters', 'Expose skin to friction or heat', 'Ignore signs of infection']
    },
    'Atopic Dermatitis': {
        'do': ['Moisturize regularly', 'Use doctor-recommended treatments', 'Identify and avoid triggers'],
        'dont': ['Scratch or rub the skin', 'Ignore flare-ups', 'Use scented or harsh products']
    },
    'Actinic Keratosis,Basal Cell Carcinoma and other Malignant Lesions': {
        'do': ['Wear sunscreen daily', 'Schedule regular skin check-ups', 'Seek prompt treatment for suspicious lesions'],
        'dont': ['Ignore skin changes', 'Use tanning beds', 'Delay medical consultation']
    },
    'Acne and Rosacea': {
        'do': ['Cleanse your face with a mild cleanser', 'Use non-comedogenic products', 'Follow dermatologist-recommended treatments'],
        'dont': ['Pick or pop pimples', 'Use harsh scrubs or products', 'Ignore worsening acne']
    },
    'Vascular Tumors': {
        'do': ['Monitor any skin changes', 'Seek medical evaluation', 'Follow prescribed treatments'],
        'dont': ['Ignore rapid growth or pain', 'Delay diagnosis', 'Self-diagnose without a doctors opinion']
    },
    'Vasculitis Disease': {
        'do': ['Follow your doctors medication plan', 'Stay hydrated', 'Report any new symptoms to your doctor'],
        'dont': ['Ignore persistent pain or rashes', 'Self-medicate without advice', 'Delay treatment for worsening symptoms']
    }


    # Add similar entries for all diseases...
}
app.secret_key = 'Pap200303'  # Secret key for session management
ADMIN_USERNAME = "Preeti"
ADMIN_PASSCODE = "1234"  # Change this to your desired passcode

# ----------------  (Welcome) Page ----------------
@app.route('/')
def welcome():
    language = session.get('language', 'en')  # Default to English
    return render_template('welcome.html', language=language)


# ---------------- Language Selection Route ----------------
@app.route('/set_language', methods=['POST'])
def set_language():
    session['language'] = request.form['language']  # Store language in session
    return "OK", 200  # Return a success response

# ---------------- Patient (Redirects to login) ----------------
@app.route('/patient')
def patient():
    return redirect('/login')  # Redirect to login page (login.html)

# ---------------- Admin Login Page ----------------
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        passcode = request.form.get('passcode')

        if username == ADMIN_USERNAME and passcode == ADMIN_PASSCODE:
            session['admin_authenticated'] = True  # Store session for authentication
            return redirect('/patients')  # Redirect to patients list page
        else:
            return "Incorrect username or passcode! Please try again."

    return render_template('admin_login.html')  # Render login input page

# Initialize database
def init_db():
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT NOT NULL,
        disease TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

# Preprocess image function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension


 #---------------------- Patient login page---------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        file = request.files.get('file')

        if not name or not email or not phone or not file:
            return "Please fill in all fields and upload an image."

        try:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            disease = classes[class_index]

            # Save patient data in the database
            try:
                conn = sqlite3.connect('patients.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO patients (name, email, phone, disease) VALUES (?, ?, ?, ?)", 
                               (name, email, phone, disease))
                conn.commit()
            except Exception as e:
                print("Database Error:", e)
                return "Error saving data to the database."
            finally:
                conn.close()

            # Fetch disease guidelines
            guidelines = disease_guidelines.get(disease, {'do': ["No specific 'Do' advice provided."], 
                                                          'dont': ["No specific 'Don't' advice provided."]})

            # Convert `do` and `dont` lists into individual query parameters
            query_params = {'name': name, 'disease': disease}
            for i, item in enumerate(guidelines['do']):
                query_params[f'do_{i}'] = item
            for i, item in enumerate(guidelines['dont']):
                query_params[f'dont_{i}'] = item

            # Corrected `url_for` call
            return redirect(url_for('result', **query_params))
        except Exception as e:
            print("Image Processing or Prediction Error:", e)
            return "Error processing the image or making a prediction."

    language = session.get('language', 'en')
    return render_template('login.html', language=language)

# ---------------- Result Page (Displays Patient Prediction) ----------------
@app.route('/result', methods=['GET'])
def result():
    name = request.args.get('name')
    disease = request.args.get('disease')
    
    # Fetch guidelines from the disease_guidelines dictionary
    guidelines = disease_guidelines.get(disease, {'do': ["No specific 'Do' advice provided."], 
                                                  'dont': ["No specific 'Don't' advice provided."]})
    do = guidelines['do']
    dont = guidelines['dont']

    language = session.get('language', 'en')
    return render_template('result.html', name=name, disease=disease, do=do, dont=dont, language=language)


# ---------------- Patients List Page (Admin View with Authentication) ----------------
@app.route('/patients')
def patients():
    if not session.get('admin_authenticated'):
        return redirect('/admin')  # Redirect back to passcode input if not authenticated

    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, disease FROM patients")
    patients = cursor.fetchall()
    conn.close()
    
    return render_template('patients.html', patients=patients)



# Route to delete all records in the 'patients' table
#@app.route('/delete_all', methods=['GET'])
#def delete_all():
    try:
        # Connect to the database
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()

        # Delete all records from the 'patients' table
        cursor.execute("DELETE FROM patients")

        # Reset the autoincrement counter
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='patients'")

        # Commit the changes
        conn.commit()

        return "All records deleted and ID counter reset successfully!"

    except Exception as e:
        print("Error:", e)
        return "Error occurred while deleting records."

    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)