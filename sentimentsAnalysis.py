#Used for image and video analysis
import cv2 

#Used for sentiment analysis
from deepface import DeepFace

#Used for image captioning
import requests

#Used for caption analysis
from textblob import TextBlob

#Used for cleaning directories and Removing temperory files
from os import remove as removeFile


def imageAnalysis(img):
    """
    Image analysis for sentiments.
    Takes: Path of Image and processes using DeepFace for sentiments,
    Gives: Some most fixed and frequent emotions along with a message with most dominant emotion.
    """
    try:
        #Analyzing
        img = cv2.imread(img)
        # genCaption = imageCaptioning(img) 
        result = DeepFace.analyze(img, actions = ['emotion'], enforce_detection=False)
        # genCaption = 'proof'

        #Storing values
        neutral = round(result[0]['emotion']['neutral'],3)
        angry = round(result[0]['emotion']['angry'],3)
        happy = round(result[0]['emotion']['happy'],3)
        sad = round(result[0]['emotion']['sad'],3)
        surprise = round(result[0]['emotion']['surprise'],3)
        
        #Giving the most dominant emotion as a message
        dominant = result[0]['dominant_emotion']

        data = {
            "neutral" : neutral,
            "angry" : angry,
            "happy" : happy,
            "sad" : sad,
            "surprise" : surprise,
            "message" : f"Most Dominant/Significant Emotion is <b>{dominant.capitalize()}</b>"+'\n'+"\n Image Description :\n"
        }
        
        return data
    
    except Exception as e:
        return {
            "neutral" : 0,
            "angry" : 0,
            "happy" : 0,
            "sad" : 0,
            "surprise" : 0,
            "message" : f"Error:{e}"
        }
    
def captionAnalysis(caption):
    """
    Caption analysis for sentiments.
    Polarity = [0,1]
    Input: Text/Caption 
    Output: A message with Very Positive, Positive, Neutral, Negative, Very Negative
    """

    blob = TextBlob(caption)

    # You can access the sentiment polarity and subjectivity of the text
    polarity = blob.sentiment.polarity

    if 0.5 > polarity > 0 :
        sentiment = "Positive"

    elif polarity >= 0.5:
        sentiment = "Very Positive"

    elif -0.5 < polarity < 0:
        sentiment = "Negative"

    elif polarity <= -0.5:
        sentiment = "Very Negative"

    else:
        sentiment = "Neutral"

    return f"\nSentiment of Caption is <b>{sentiment}</b>: <b> {polarity*100}</b>%"

def imageCaptioning(imgPath):
    """
    Uses API and image path for Image captioning.
    Input: Image Path
    Output: Image Captions in a Line
    """

    API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    headers = {"Authorization": "Bearer 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"}

    with open(imgPath, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)

    print(response.json()[0]['generated_text'])
    return response.json()[0]['generated_text']


def videoAnalysis(video):
    """
    Do Video sentiment Analysis using Multiple layered deepface
    Input: Video path
    Output: Gives message along with video description using imagecaptioning
    """

    # Open video capture
    cap = cv2.VideoCapture(video)

    # Get frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Variables to store overall emotion scores
    total_emotion_scores = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    total_frames = 0
    happeningOfVideo = ""

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_path = "temp_frame.jpg"

        cv2.imwrite(image_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        captionImg = imageCaptioning(image_path)

        if captionImg in happeningOfVideo:
            continue

        elif happeningOfVideo == "":
            happeningOfVideo += captionImg

        else:
            happeningOfVideo += 'and '
            happeningOfVideo += captionImg

        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

        # Extract the emotion
        emotion = result[0]['emotion']

        # Update overall emotion scores
        for emotion_type, score in emotion.items():
            total_emotion_scores[emotion_type] += score

        total_frames += 1

        # Skip frames based on the desired interval (1 frame per 5 seconds)
        skip_frames = int(fps * 5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)

    # Calculate average emotion scores
    average_emotion_scores = {emotion_type: round((total_score / total_frames),2) for emotion_type, total_score in total_emotion_scores.items()}

    # Release the video capture object
    cap.release()

    # Remove the temporary image file
    removeFile(image_path)
    # removeFile(r"static\video.mp4")

    return average_emotion_scores, happeningOfVideo

#-----For Testing Purposes-----

# print(videoAnalysis(r"C:\Users\Anubhav Choubey\Downloads\Why You Cant Code.mp4"))


# print(imageAnalysis(r"C:\Users\Anubhav Choubey\Pictures\Documents\0201AI221014.jpg"))


# print(imageCaptioning(r"C:\Users\Anubhav Choubey\Pictures\Documents\0201AI221014.jpg"))