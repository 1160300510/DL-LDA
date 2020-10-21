select distinct * from Posts where Id<=63760928 and
(Tags like '%<speech-recognition>%' or Tags like '%<speech-to-text>%' or Tags like '%<google-speech-api>%' or Tags like '%<voice-recognition>%'
or Tags like '%<cmusphinx>%' or Tags like '%<sphinx4>%' or Tags like '%<speech>%' or Tags like '%<pocketsphinx>%'
or Tags like '%<sapi>%' or Tags like '%<speech-synthesis>%' or Tags like '%<google-cloud-speech>%' or Tags like '%<mfcc>%')
and CreationDate>'2012-01-01';