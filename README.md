# Data Source

The entire dataset disclosure application is in progress and will be released later.


# Data Format

### user_behavior_history.csv
The data set covers all behaviors (including ID, call time, status, and log) of approximately 30 million random users from January 1, 2023 to June 30, 2024. The organization of the TelM-1 dataset and the TelM-2 dataset is similar. Each sample in the original log data contains the User ID, call time, connection status, call duration, etc. The detailed description of each column in the data set is as follows:

| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Call time | An integer, the timestamp of the behavior |
| Call status | A string, enum-type from ('answer', 'unanswer') |
| Call duration | A float, the call duration of the behavior |


### user_profile.csv
These two data sets mainly contains the basic attributes of about 30 million random users, such as age, gender, occupation, resident city ID, etc. Each row of the data set represents a piece of user information, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Age | An integer, the serialized ID that represents an user age |
| Gender | An integer, the serialized ID that represents the user gender |
| Occupation | An integer, the serialized ID that represents the user occupation |
| Habitual City | An integer, the serialized ID that represents the user habitual city，single value |

### historical_connection_profile.csv
This data file mainly contains the basic attribute characteristics of about 270,000 products, such as product category ID, product city, product label, etc. Each row of the data set represents the information of a product, separated by commas. The detailed description of each column in the data set is as follows:e (e.g., call times, connection times, and call duration). 
| Field | Explanation |
| --- | --- |
| Item ID | An integer, the serialized ID that represents an item |
| Category ID | An integer, the serialized ID that represents an item category |
| Item City | An integer, the serialized ID that represents an item City |
| Item Tag List | A String, the ID of each tag is separated by English semicolon after serialization |
