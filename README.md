# Data Source

The entire dataset disclosure application is in progress and will be released later.


# Data Format

### user_behavior_history.csv
The data set covers all behaviors (including clicks, favorites, adding, and purchases) of approximately 17 million random users from June 3, 2019 to June 3, 2021. The organization of the data set is similar to MovieLens-20M, namely Each row of the collection represents a user data behavior, which is composed of user ID, call_time, call_status, call_duration. Each column summarizes the detailed information of the product under investigation.
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| call_time | An integer, the timestamp of the behavior |
| call_status | A string, enum-type from ('answer', 'unanswer') |
| call_duration | An float, the call duration of the behavior |


### user_profile.csv
This data set mainly contains the basic attributes of about five million random users, such as age, gender, occupation, resident city ID, crowd tag, etc. Each row of the data set represents a piece of user information, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Age | An integer, the serialized ID that represents an user age |
| Gender | An integer, the serialized ID that represents the user gender |
| Occupation | An integer, the serialized ID that represents the user occupation |
| Habitual City | An integer, the serialized ID that represents the user habitual city，single value |
| User Tag List | A String, the ID of each tag is separated by English semicolon after serialization |

### item_profile.csv
This data file mainly contains the basic attribute characteristics of about 270,000 products, such as product category ID, product city, product label, etc. Each row of the data set represents the information of a product, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| Item ID | An integer, the serialized ID that represents an item |
| Category ID | An integer, the serialized ID that represents an item category |
| Item City | An integer, the serialized ID that represents an item City |
| Item Tag List | A String, the ID of each tag is separated by English semicolon after serialization |
