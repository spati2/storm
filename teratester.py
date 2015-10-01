import os,sys,inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"Problems/tera")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)


def test_wei():
    from tera_dataset_RF import *
    data = [antRF(), camelRF(), ivyRF(), jeditRF(), luceneRF(), poiRF(), synapseRF(), velocityRF(), xercesRF()]
    w = [[8.0, 2.0, 13.0, 132.0, 0.02, 0.92], [1.0, 8.0, 27.0, 150.0, 1.0, 0.01], [5.0, 4.0, 16.0, 129.0, 0.52, 0.8], [14.0, 2.0, 36.0, 72.0, 0.01, 1.0], [3.0, 5.0, 15.0, 68.0, 0.01, 0.37], [11.0, 7.0, 35.0, 141.0, 0.51, 1.0], [18.0, 3.0, 37.0, 125.0, 0.88, 0.01], [6.0, 14.0, 41.0, 92.0, 0.2, 0.15], [16.0, 4.0, 39.0, 97.0, 0.71, 1.0]]

    print len(data)
    print len(w)
    for d,i in zip(data,w):
        a = d
        input = i
        output = a.test(input=input)
        print a.name, output


if __name__ == "__main__":
    test_wei()


# from tera_dataset import *

# a = camel()

# input = []
# for i,arg in enumerate(sys.argv):

#     if arg is not None and i > 0:
#         input.append(arg)
# # <mss>8.8641246794</mss>
# # <msl>14.9794761475</msl>
# # <mxf>0.929123322772</mxf>
# # <threshold>0.915820856764</threshold>


# if len(input) == 0:
#     input = [8.8641246794, 14.9794761475, 0.929123322772, 5, 0.915820856764]
#     input_wei = [13, 6, 0.01, 38, 0.33]

#     mss = int(round(input[0]))
#     msl  = int(round(input[1]))
#     mxf = float(input[2])
#     thres = float(input[3])
#     #cri = "mse"

#     output = a.test(input=input)
#     default = a.default()
#     #output = [float(str("%.2f" % out)) for out in output]

#     output2 = a.test(input=input_wei)
#     #print [mss, msl, mxf, thres], output
#     print "Vivek: ", output
#     print "Default: ", default
#     print "Wei: ", output2
# else:
#     print "invalid input"