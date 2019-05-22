# Ancient Coin Identification

This web app is a tool that helps users identify and assess the value of ancient Roman coins. A machine learning model is used to identify the portrait of the emperor appearing on the coin, and users are provided with an interactive visualization of ancient coin auction prices for that emperor.

The app is developed with Python, Keras for Machine Learning, Flask, HTML/CSS, and Bokeh for visualizations, and deployed on Heroku.

## Web App URL
[https://xyz.herokuapp.com/](https://xyz.herokuapp.com/)

## How It Works
This web application is designed to help users identify and assess the value of ancient Roman coins. Users can link to or upload an image of a Roman coin, and a machine learning model will process the image and predict which emperor appears on the coin based on the distinct features of the obverse (frontside) portrait. In addition, users are provided with an interactive visualization comprised of past coin auction data that provides a way to assess the value of coins specific to a particular emperor, based on denomination, grade, and other features. The idea is to provide everyone from new collectors to enthusiasts with information to help them learn more about ancient coins and better assess their value.

### Machine Learning Model
The portrait identification model is implemented as a 3-layer [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) using the Keras framework. The model is trained with over 8,200 images of imperial Roman coins, corresponding to 9 different Roman emperors, which were downloaded from several online coin auction websites using the Requests and Beautiful Soup packages. The dataset consists of roughly between 600 to 1200 images per emperor, and the images are standardized for analysis (rescaled and converted to grayscale) using OpenCV. Image augmentation is performed using the Keras API to artificially boost the dataset for model training, and the overall accuracy is approximately 95% on out-of-sample test data.

### Data Visualization
Along with coin images, coin auction lot descriptions are scraped as well, and relevant keywords are extracted to form a database. Each coin in the database is identified by several features, such as the auction in which it was listed, the auction lot number, which portrait appears on the obverse, the denomination, the price realized, its grade, and whether or not it's toned, centered, etc. The Bokeh plotting package is used to construct an interactive visualization of this data in the form of a scatter plot of sale prices realized. The user can apply different selection criteria to see how the sale prices evolve, providing an estimate of value.

### Future Plans
In addition to portrait identification, I'd like to extend this app to extract the inscriptions appearing around the perimeter of the coin, which together with the portrait would allow for a complete identification via the [RIC](https://en.wikipedia.org/wiki/Roman_Imperial_Coinage) number.