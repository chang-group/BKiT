from setuptools import setup
import os
import sys
import sysconfig

if __name__=='__main__':
	Packages = ['BKit']
	
	setup(	name='BKit',
			version="0.1",
			packages = Packages,
			author="Talant Ruzmetov, ..." ,
			author_email="talantruzmetov@gmail.com",
			description="This is accurate free energy calculation toolkit to charactirize molecular transition",
			license="MIT",
			keywords="free energy, PCA, protein, ligand, kinetics, milestoning",
			url="https://github.com/truzmeto/MileStoningKit",
			project_urls={
				"Bug Tracker": "https://github.com/truzmeto/MileStoningKit/issues",
				"Documentation": "https://github.com/truzmeto/MileStoningKit/tree/Release/Doc",
				"Source Code": "https://github.com/truzmeto/MileStoningKit",
			})
