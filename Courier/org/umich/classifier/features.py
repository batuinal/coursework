#vocab = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'course', 'class', 'syllabus', 'handout', 'homework', 'cs', 'lecture', 'notes', 'slides', 'solution', 'problem', 'program', 'instructor', 'information', 'project', 'paper', 'guide', 'study', 'prelim', 'professional', 'activities', 'resume', 'publications', 'language', 'research', 'teaching', 'contact', 'projects', 'professor', 'interests', 'department', 'personal', 'office', 'advisor', 'home', 'page', 'links', 'phone']
vocab = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'course', 'class', 'syllabus', 'handout', 'homework', 'lecture', 'notes', 'slides', 'solution', 'problem', 'program', 'instructor', 'information', 'project', 'paper', 'guide', 'study', 'activities', 'projects', 'professor', 'office']
relevantURLsPattern = ['slide', 'handout', 'schedule', 'syllabus', 'homework', 'lecture', 'assignment', 'project', 'exam', 'midterm', 'final', 'notes', 'staff', 'hours', 'course-info', 'piazza']
relevantTags = ['li', 'ul', 'a', 'h1', 'h2', 'h3'];
stopWords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your']
numericCharacterCount = 'numericCharCount'
relevantURLsCount = 'URLCount'
relevantTagsCount = 'TagCount'
documentLength = 'docLength'
docTitle = 'docTitle'
anchorCount = 'anchorTagCount'
listCount = 'listTagCount'
paragraphCount = 'paraTagCount'
headerCount = 'headerTagCount'

#metadataKeys = [anchorCount, listCount, paragraphCount, headerCount, numericCharacterCount, relevantURLsCount, relevantTagsCount, documentLength, docTitle]
metadataKeys = [anchorCount, listCount, paragraphCount, headerCount, numericCharacterCount, documentLength, docTitle]