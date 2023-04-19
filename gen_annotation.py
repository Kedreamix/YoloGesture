from lxml import etree
 
class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
 
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
 
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
 
        child3 = etree.SubElement(self.root, "source")
 
        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child5.text = "Unknown"
 
##        child6 = etree.SubElement(child3, "image")
##        child6.text = "flickr"
##        child7 = etree.SubElement(child3, "flickrid")
##        child7.text = "35435"
 
 
    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)
    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')
    def add_pic_attr(self,label,xmin,ymin,xmax,ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)
 
 
if __name__ == '__main__':
    filename="000001.jpg"
    anno= GEN_Annotations(filename)
    anno.set_size(1280,720,3)
    for i in range(3):
        xmin=i+1
        ymin=i+10
        xmax=i+100
        ymax=i+100
        anno.add_pic_attr("pikachu",xmin,ymin,xmax,ymax)
    anno.savefile("00001.xml")
