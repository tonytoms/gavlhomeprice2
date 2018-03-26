from datetime import datetime
from datetime import date


import holidays
import calendar

datetime_object = datetime.strptime('25/03/2018', '%d/%m/%Y')


day=(calendar.day_name[datetime_object.weekday()])
timeofWeek=''
if(day=="Sunday" or day=="Saturday"):
    timeofWeek="WeekEnd"
else:
    timeofWeek="WeekDay" 
 

aus_holidays = holidays.AU(prov = 'VIC')  # or holidays.US(), or holidays.CountryHoliday('US')

if date(2018, 4, 2) in aus_holidays:
    print('yes')