import os
from bs4 import BeautifulSoup
import plotly.express as px


def parse_bookmarks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    # Start from the top-level <DL>
    dl = soup.find('dl')
    return parse_dl(dl)

def parse_dl(dl_tag):
    """
    Recursively parse a <DL> tag, returning a list of folder/link dicts.
    """
    result = []
    if not dl_tag:
        return result
    
    dt_tags = dl_tag.find_all('dt', recursive=False)
    for dt in dt_tags:
        h3 = dt.find('h3', recursive=False)
        if h3:
            # This is a folder/heading
            folder_name = h3.get_text(strip=True)
            # The next sibling DL contains its contents
            sub_dl = dt.find_next_sibling('dl')
            children = parse_dl(sub_dl)
            # Count all links in this folder and subfolders
            link_count = sum(child['link_count'] for child in children)
            # Also count direct links under this folder
            direct_links = [a for a in dt.find_all('a', recursive=False)]
            link_count += len(direct_links)
            result.append({
                'name': folder_name,
                'link_count': link_count,
                'children': children
            })
        else:
            # This is a direct link (not in a folder)
            a = dt.find('a', href=True)
            if a:
                result.append({
                    'name': a.get_text(strip=True),
                    'link_count': 1,
                    'children': []
                })
    return result

def flatten_tree_for_sunburst(tree, parent_name=None):
    """
    Flatten the tree into lists for sunburst: ids, parents, values.
    """
    ids = []
    parents = []
    values = []
    def _recurse(node, parent):
        ids.append(node['name'])
        parents.append(parent if parent else "")
        values.append(node['link_count'])
        for child in node.get('children', []):
            _recurse(child, node['name'])
    for node in tree:
        _recurse(node, parent_name)
    return ids, parents, values

def visualize_bookmark_counts(tree, output_file='bookmark_sunburst.html'):
    ids, parents, values = flatten_tree_for_sunburst(tree)
    print(f"Number of nodes: {len(ids)}")
    print(f"Sample nodes: {list(zip(ids, parents, values))[:5]}")
    fig = px.sunburst(
        ids=ids,
        parents=parents,
        values=values,
        title="Bookmark Folder Link Counts"
    )
    fig.write_html(output_file)
    fig.show()
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    bookmark_file = os.path.join(os.path.dirname(__file__), 'bookmarks_24_04_2025.html')
    tree = parse_bookmarks(bookmark_file)
    visualize_bookmark_counts(tree)
