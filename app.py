import streamlit as st

st.title('Streamlit Demo')
st.header('First Heading')
st.subheader('sub-heading')
st.text('Example Text')
st.success('Hurray')
st.warning('error')
st.info('info')
st.error('error')

if st.checkbox('select'):
    st.info('checkbox is selected')
else:
    st.text('No Checkbox is selected')

state=st.radio("what is favourite colour",{'red','green','blue'})

occupation=st.selectbox('What is your Occupation',('student','govt. officer','private','Bussiness man'))

if occupation=='student':
    st.text('i am also student')

if st.button('click'):
    st.error('button is clicked')


st.sidebar.header('Neeraj Singh')
st.sidebar.text('ML Student')