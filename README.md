# Colorify

### Live Demo

[project.colorify.link](http://project.colorify.link)

## University Name: [San Jose State University](http://www.sjsu.edu/)

## Course: [Enterprise Software CMPE-172](http://info.sjsu.edu/web-dbgen/catalog/courses/CMPE172.html)

## Professor: [Sanjay Garje](https://www.linkedin.com/in/sanjaygarje/)

## Student: 
  ###   [Manpreet Singh](https://www.linkedin.com/in/manpreet-singh96)
  ###   [Manmeet Gill](https://www.linkedin.com/in/manmeet-gill1/)
  ###   [Vicson Moses](https://www.linkedin.com/in/vicson-moses)


## Project Introduction (What the application does, feature list)
  `The user is able to easily navigate our webpage`

  `The user is able to upload a grayscale image`

  `The user is able to colorify his grayscale image`

  `The user is able to store the colored image by downloading`

## Sample Demo Screenshots

![Albert Einstein](https://github.com/mannysinghh11/Colorify/blob/master/demos/albert.jpg "Colorized image of Sir Albert Einstein")

![Barack Obama](https://github.com/mannysinghh11/Colorify/blob/master/demos/obama.jpg "Colorized image of Barack Obama")


## Pre-requisites Set Up

### AWS
   * `S3 bucket`
 
   * `Lambda Function`
 
   ![Lambda Function](https://github.com/mannysinghh11/Colorify/blob/master/demos/lambdafunction.jpg "Lambda Function")
 
      make sure to update the 'Bucket' and 'Key' in the params variable to point to your S3 bucket.

      make sure to also update env variable with your aws access and secret keys

  * `EC2 (for deployment)`
 
  * `Route53 (for routing)`
 ### AWS Architecture Diagram
![AWS Architecture Diagram](https://github.com/mannysinghh11/Colorify/blob/master/demos/aws.jpeg "AWS Architecture Diagram")

### Application and model dependencies

  * `Python (use pip to install the libraries listed below`

  * `flask`

  * `numpy`

  * `keras`

  * `Sci-py`

  * `imageio`

  * `scikit-image`

  * `boto3`

After every requirement is installed, run the python file by using the following commands:

`   from app directory` run `python app.py`

Open any browser and open the following URL to use the application

`   http://localhost:80`

### References

`https://www.codementor.io/dushyantbgs/deploying-a-flask-application-to-aws-gnva38cf0`

`https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2`

`https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/`



