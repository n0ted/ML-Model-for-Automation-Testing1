from bs4 import BeautifulSoup
import pandas as pd

# Assuming html_content contains your HTML
with open('c:/Users/Lenovo/Downloads/Test/htmltest.html', 'a', encoding='utf-8') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

data = []

# Function to extract features for any given element
def extract_features(element, element_name):
    return {
        'Element': element_name,
        'TagName': element.name,
        'Class': ' '.join(element.get('class', '')),
        'TextLength': len(element.text.strip()),
        'HasChildren': len(element.find_all(recursive=False)) > 0,
        'Depth': len(element.find_parents())
    }

# List of element types and their classes to iterate over
elements_to_find = [
    {'class': 'header', 'name': 'Header'},
    {'class': 'navbar', 'name': 'Navbar Link'},
    {'class': 'main', 'name': 'Main Content Heading'},
    {'class': 'footer', 'name': 'Footer'}
]

for element in elements_to_find:
    for found_element in soup.find_all(class_=element['class']):
        data.append(extract_features(found_element, element['name']))

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('annotated_dataset.csv', index=False)
