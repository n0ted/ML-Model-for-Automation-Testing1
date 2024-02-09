from bs4 import BeautifulSoup
import pandas as pd

# Sample HTML data
with open('ML Training/DOM Files/htmltest.html', 'r',encoding = 'utf-8') as file:
    html_content = file.read()
# Parse the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Prepare a list to hold our data
data = []

# By ID
for element in soup.find_all(id=True):
    data.append({
        'locator_type': 'id',
        'locator_value': element['id'],
        'tag_name': element.name,
        'text': element.text.strip()
    })

# By Class Name
for element in soup.find_all(class_=True):
    for class_name in element['class']:
        data.append({
            'locator_type': 'class_name',
            'locator_value': class_name,
            'tag_name': element.name,
            'text': element.text.strip()
        })

# By Name Attribute
for element in soup.find_all(attrs={"name": True}):
    data.append({
        'locator_type': 'name',
        'locator_value': element['name'],
        'tag_name': element.name,
        'text': element.text.strip()
    })

# By Tag Name (demonstration with 'a' tag)
for element in soup.find_all('a'):
    data.append({
        'locator_type': 'tag_name',
        'locator_value': element.name,
        'tag_name': element.name,
        'text': element.text.strip()
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('html_elements_locators.csv', index=False)

print(df)
