import pywikibot
from pywikibot import pagegenerators
import time

#test using wikibot to retrieve entity info + wikipedia url

def get_wikipedia_page(item, language='en'):
    if language + 'wiki' in item.sitelinks:
        sitelink = item.sitelinks[language + 'wiki']
        wiki_site = pywikibot.Site(language, "wikipedia")
        wiki_page = pywikibot.Page(wiki_site, sitelink.title)
        return wiki_page
    else:
        return None

def get_biographical_info(item_id, language='en'):
    site = pywikibot.Site("wikidata", "wikidata")
    repo = site.data_repository()
    item = pywikibot.ItemPage(repo, item_id)
    item.get()
    
    bio_info = {
        'name': item.labels.get('en', 'Unknown'),
        'description': item.descriptions.get('en', 'No description available'),
        'birth_date': None,
        'death_date': None,
        'occupations': [],
        'notable_works': [],
        'wikipedia_url': None
    }
    
    BIRTH_DATE = 'P569'
    DEATH_DATE = 'P570'
    OCCUPATION = 'P106'
    NOTABLE_WORK = 'P800'
    
    if BIRTH_DATE in item.claims:
        bio_info['birth_date'] = item.claims[BIRTH_DATE][0].getTarget().year
    
    if DEATH_DATE in item.claims:
        bio_info['death_date'] = item.claims[DEATH_DATE][0].getTarget().year
    
    if OCCUPATION in item.claims:
        for claim in item.claims[OCCUPATION]:
            occupation = claim.getTarget()
            bio_info['occupations'].append(occupation.labels.get('en', 'Unknown occupation'))
    
    if NOTABLE_WORK in item.claims:
        for claim in item.claims[NOTABLE_WORK]:
            work = claim.getTarget()
            bio_info['notable_works'].append(work.labels.get('en', 'Unknown work'))
    
    wiki_page = get_wikipedia_page(item, language)
    if wiki_page:
        bio_info['wikipedia_url'] = wiki_page.full_url()
    
    return bio_info

def main():
    start_time = time.time() 
    
    wikidata_id = 'Q937'  # Albert Einstein's Wikidata ID
    language = 'en'  
    bio_info = get_biographical_info(wikidata_id, language)
    
    print(f"Name: {bio_info['name']}")
    print(f"Description: {bio_info['description']}")
    print(f"Birth Date: {bio_info['birth_date']}")
    print(f"Death Date: {bio_info['death_date']}")
    print("Occupations:")
    for occupation in bio_info['occupations']:
        print(f"  - {occupation}")
    print("Notable Works:")
    for work in bio_info['notable_works']:
        print(f"  - {work}")
    print(f"Wikipedia URL: {bio_info['wikipedia_url']}")
    
    end_time = time.time()  # End timing
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()