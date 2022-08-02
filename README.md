# Emotion-Based Music Recommendation System - SERVER

## Contribution

### For Collaborators

- Create a new branch
- Commit the fix to the newly created branch
- Push to that branch for review
- Merge after the code has been reviewed by another member

        git checkout -b <new-branch-name>
        git pull
        git add .
        git commit -m "<message-for-the-commit>"
        git push origin <new-branch-name>

### For First-time Contributors of the Repo

- Install all packages
  pip install -r requirements.txt

- Follow the same process as the Collaborators in the previous section

## Experimenting

- To start the flask server, stay on the main directory and run

        python3 main.py

- Will include a testing file later to test whether the server is running

## Testing with Docker

- Build image and run

        docker build -t eaai23 . && docker run -d -p 8080:8080 -t eaai23

- Now the server is hosting at http://localhost:8080

## Publishing image to Cloud Registry and deploy to Cloud Run

- Install [**GCloud CLI**](https://cloud.google.com/sdk/docs/install) (if haven't)
- Login to your [Google Cloud Email](https://accounts.google.com/signin/v2/identifier?service=cloudconsole&passive=1209600&osid=1&continue=https%3A%2F%2Fconsole.cloud.google.com%2F&followup=https%3A%2F%2Fconsole.cloud.google.com%2F&flowName=GlifWebSignIn&flowEntry=ServiceLogin) (if haven't)

- Set the current project to **\<Your-project-name-on-the-cloud\>**

        gcloud config set project <Your-project-name-on-the-cloud>

- Submit image to Cloud Registry

        gcloud builds submit --tag gcr.io/<Your-project-name-on-the-cloud>/<your-instance-name> --timeout=2h15m5s

- Deploy on Cloud Run
  - Go to Cloud Run
  - Click on **<your-instance-name>** instance
  - Click on **EDIT & DEPLOY NEW VERSION**
  - Select a new Container image URL
  - Deploy
