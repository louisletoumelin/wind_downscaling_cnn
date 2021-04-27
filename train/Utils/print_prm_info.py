def print_prm_information(prm):
    print('\n\n__________________________\n')
    print('_____Prm configuration____\n')
    print('__________________________\n\n')
    print('Parameters: \n')
    print(prm)
    print('\n')


def print_finish():
    print('\n\n__________________________')
    print('__________________________\n')
    print('_______prm finished_______\n')
    print('__________________________')
    print('__________________________\n\n')


def print_end_of_job():
    print('\n\n__________________________')
    print('\n__________________________\n')
    print('__________________________\n')
    print('________End of job________\n')
    print('__________________________')
    print('\n__________________________\n')
    print('__________________________\n\n')


def print_intro():

    intro = """
                                            ''' '
                                          '   ' '
                                     ''' ''' '''
                              + hs    ' '''''  '.' '   
                            'shh  ho            '   '   
                           .yhhh  hh+           ' ''  
                          /hhhs    +hhh/         
                          hhhh'     hhhh         '''
                         ohhho      +hhh:       '.  '.' 
                       'yhhh:        ohhh: ''''' ''' .  
               .+.    -hhhy.          ohhh:  '  ''''' ''
              -hhho' /hhhs'            ohhh:  ''''''''' 
             :hhhhhhyhhh+               ohhh/      .' ''
            /hhho+hhhhh:                 +hhh+    '. '.'
           +hhh+  '+hy                    /hhho     ''  
          ohhh/     '                       :hhhs'       
        'shhh:                               :yhhy-      
       gyhhhg           Wind speed            'shhh/     
      hyhhyf                                   +hhhs'   
     :hhhs'             Downscaling              -hhhh:  
    +hhho                                       'ohhhsg
    hhh/                 using CNN                  :yhhh
    hy-                                              '+hh
    o'              by Louis Le Toumelin               .s

                     CEN - Meteo-France
    """
    print(intro)


def print_end_of_training():
    string = """
  ______           _          __   _             _       _             
 |  ____|         | |        / _| | |           (_)     (_)            
 | |__   _ __   __| |   ___ | |_  | |_ _ __ __ _ _ _ __  _ _ __   __ _ 
 |  __| | '_ \ / _` |  / _ \|  _| | __| '__/ _` | | '_ \| | '_ \ / _` |
 | |____| | | | (_| | | (_) | |   | |_| | | (_| | | | | | | | | | (_| |
 |______|_| |_|\__,_|  \___/|_|    \__|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                                                  __/ |
                                                                 |___/ 
                                                                 """
    print(string)